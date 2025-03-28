import argparse
import os
import torch
from torch.optim import AdamW
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from src.data_processing import load_data, create_data_loaders, get_class_names
from src.model import SentimentClassifier, train_epoch, eval_model, save_model
from src.evaluation import compute_metrics, plot_confusion_matrix, plot_metrics_history, analyze_misclassifications, save_evaluation_results

def main(args):
    """
    Main entry point for the Amazon Review Sentiment Analysis pipeline
    
    This function orchestrates the entire process from data loading to model 
    training and evaluation based on command-line arguments.
    """
    # Create necessary directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Load data
    print("Loading data...")
    train_df, test_df = load_data(
        file_path=args.data_path,
        use_huggingface=args.use_huggingface,
        sample_size=args.sample_size
    )
    
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_df, 
        test_df, 
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Initialize model
    model = SentimentClassifier(n_classes=3)
    model = model.to(device)
    
    # Train model if specified
    if args.train:
        print("\n" + "="*50)
        print("TRAINING MODEL")
        print("="*50)
        
        # Prepare optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        total_steps = len(train_loader) * args.epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        best_accuracy = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
            
            # Train
            train_loss, train_acc = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                device, 
                scheduler
            )
            
            # Validate
            val_loss, val_acc, _, _ = eval_model(
                model,
                val_loader,
                device
            )
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Update training history
            training_history['train_loss'].append(train_loss)
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss)
            training_history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
                print(f"Saved best model with accuracy: {best_accuracy:.4f}")
        
        # Save the final model and tokenizer
        save_model(model, tokenizer, args.output_dir)
        
        # Plot training metrics
        history_fig = plot_metrics_history(training_history)
        history_fig.savefig(os.path.join(args.output_dir, "results", "training_history.png"))
        
        print("\nTraining complete!")
    
    # Evaluate model
    if args.evaluate:
        print("\n" + "="*50)
        print("EVALUATING MODEL")
        print("="*50)
        
        # Load best model if we trained
        if args.train:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "best_model.pt")))
            print("Loaded best model from training")
        # Or load existing model if we didn't train
        elif os.path.exists(os.path.join(args.model_path, "best_model.pt")):
            model.load_state_dict(torch.load(os.path.join(args.model_path, "best_model.pt"), map_location=device))
            print(f"Loaded existing model from {args.model_path}")
        else:
            print("No model found for evaluation. Please train a model first.")
            return
        
        # Evaluate on test set
        test_loss, test_acc, predictions, targets = eval_model(
            model,
            test_loader,
            device
        )
        
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Compute metrics
        class_names = get_class_names()
        metrics = compute_metrics(predictions, targets)
        
        # Print classification report
        report = metrics['classification_report']
        print("\nClassification Report:")
        for label, metrics_dict in report.items():
            if isinstance(metrics_dict, dict):
                print(f"{label}:")
                for metric_name, value in metrics_dict.items():
                    print(f"  {metric_name}: {value:.4f}")
        
        # Create confusion matrix plot
        cm_fig = plot_confusion_matrix(metrics['confusion_matrix'], class_names)
        
        # Analyze misclassifications
        misclassified = analyze_misclassifications(model, test_loader, device, n_examples=5)
        
        # Print some misclassified examples
        print("\nMisclassified Examples:")
        for i, example in enumerate(misclassified[:3]):  # Just show first 3
            print(f"\nExample {i+1}:")
            print(f"Text: {example['text'][:100]}...")
            print(f"Predicted: {example['predicted']}, Actual: {example['actual']}")
        
        # Save evaluation results
        figures_dict = {
            'confusion_matrix': cm_fig
        }
        
        save_evaluation_results(
            {
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'classification_report': report,
                'misclassified_examples': misclassified
            },
            figures_dict,
            os.path.join(args.output_dir, "results")
        )
        
        print(f"\nEvaluation results saved to {os.path.join(args.output_dir, 'results')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Amazon Review Sentiment Analysis")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the data file")
    parser.add_argument("--use_huggingface", action="store_true", help="Use HuggingFace datasets")
    parser.add_argument("--sample_size", type=int, default=None, help="Number of samples to use")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--output_dir", type=str, default="models/sentiment", help="Output directory")
    parser.add_argument("--model_path", type=str, default="models/sentiment", help="Path to load model from")
    
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        print("Please specify at least one operation to perform: --train or --evaluate")
    else:
        main(args)