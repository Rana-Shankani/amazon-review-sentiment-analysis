import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from src.data_processing import get_class_names

def compute_metrics(predictions, targets):
    """
    Compute classification metrics
    
    Args:
        predictions: List of model predictions
        targets: List of ground truth labels
        
    Returns:
        Dictionary containing confusion matrix and classification report
    """
    class_names = get_class_names()
    
    # Calculate confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Get classification report
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
    
    return {
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_confusion_matrix(confusion_matrix, class_names, figsize=(10, 8)):
    """
    Plot the confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix from sklearn
        class_names: List of class names
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        confusion_matrix, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_metrics_history(training_history, figsize=(12, 5)):
    """
    Plot training and validation loss/accuracy over epochs
    
    Args:
        training_history: Dictionary containing training metrics
        figsize: Size of the figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot training and validation loss
    ax1.plot(training_history['train_loss'], label='Training Loss')
    ax1.plot(training_history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot training and validation accuracy
    ax2.plot(training_history['train_acc'], label='Training Accuracy')
    ax2.plot(training_history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def analyze_misclassifications(model, data_loader, device, n_examples=5):
    """
    Analyze examples where the model made incorrect predictions
    
    Args:
        model: SentimentClassifier model
        data_loader: Data loader containing examples
        device: Device to run prediction on
        n_examples: Number of misclassified examples to return
        
    Returns:
        List of misclassified examples
    """
    model.eval()
    misclassified_examples = []
    class_names = get_class_names()
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            review_texts = batch['review_text']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            
            # Find misclassified examples
            for i, (pred, target) in enumerate(zip(preds, targets)):
                if pred != target:
                    misclassified_examples.append({
                        'text': review_texts[i],
                        'predicted': class_names[pred.item()],
                        'actual': class_names[target.item()]
                    })
                    
                    if len(misclassified_examples) >= n_examples:
                        return misclassified_examples
    
    return misclassified_examples

def save_evaluation_results(metrics, figures_dict, output_dir):
    """
    Save evaluation metrics and figures to disk
    
    Args:
        metrics: Dictionary of evaluation metrics
        figures_dict: Dictionary of matplotlib figures
        output_dir: Directory to save results to
    """
    import os
    import json
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save metrics as JSON
    # Convert numpy values to Python native types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    metrics_json = {k: convert_for_json(v) for k, v in metrics.items()}
    
    with open(f"{output_dir}/metrics.json", 'w') as f:
        json.dump(metrics_json, f, indent=4)
    
    # Save figures
    for name, fig in figures_dict.items():
        fig.savefig(f"{output_dir}/{name}.png")
        plt.close(fig)