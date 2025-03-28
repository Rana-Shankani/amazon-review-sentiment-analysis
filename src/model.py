import torch
import torch.nn as nn
from transformers import DistilBertModel
import numpy as np
from tqdm import tqdm

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes=3):
        """
        Sentiment classification model using DistilBERT
        
        Args:
            n_classes: Number of sentiment classes (default: 3 - negative, neutral, positive)
        """
        super(SentimentClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.distilbert.config.hidden_size, n_classes)
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask for padding
            
        Returns:
            Logits for each sentiment class
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use the CLS token for classification
        pooled_output = outputs.last_hidden_state[:, 0]
        output = self.drop(pooled_output)
        return self.out(output)

def train_epoch(model, data_loader, optimizer, device, scheduler=None):
    """
    Train the model for one epoch
    
    Args:
        model: SentimentClassifier model
        data_loader: Training data loader
        optimizer: PyTorch optimizer
        device: Device to train on
        scheduler: Learning rate scheduler (optional)
        
    Returns:
        avg_loss, accuracy: Average loss and accuracy for the epoch
    """
    model.train()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, targets)
        
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == targets)
        total_predictions += targets.shape[0]
        
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        progress_bar.set_postfix({'loss': np.mean(losses), 'accuracy': correct_predictions.item() / total_predictions})
    
    return np.mean(losses), correct_predictions.item() / total_predictions

def eval_model(model, data_loader, device):
    """
    Evaluate the model
    
    Args:
        model: SentimentClassifier model
        data_loader: Validation or test data loader
        device: Device to evaluate on
        
    Returns:
        avg_loss, accuracy, predictions, targets: Evaluation metrics and data
    """
    model.eval()
    losses = []
    correct_predictions = 0
    total_predictions = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['targets'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets)
            
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets)
            total_predictions += targets.shape[0]
            
            losses.append(loss.item())
            
            all_predictions.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return np.mean(losses), correct_predictions.item() / total_predictions, all_predictions, all_targets

def save_model(model, tokenizer, output_dir):
    """
    Save the model and tokenizer
    
    Args:
        model: SentimentClassifier model
        tokenizer: DistilBERT tokenizer
        output_dir: Directory to save to
    """
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Saving model to {output_dir}")
    torch.save(model.state_dict(), f"{output_dir}/pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)
    
def load_model(model_path, device, n_classes=3):
    """
    Load a saved model
    
    Args:
        model_path: Path to the saved model
        device: Device to load model on
        n_classes: Number of sentiment classes
        
    Returns:
        model, tokenizer: Loaded model and tokenizer
    """
    from transformers import DistilBertTokenizer
    
    model = SentimentClassifier(n_classes=n_classes)
    model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin"))
    model = model.to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    
    return model, tokenizer

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    """
    Make a sentiment prediction for a single text
    
    Args:
        text: Review text
        model: SentimentClassifier model
        tokenizer: DistilBERT tokenizer
        device: Device to run prediction on
        max_length: Maximum sequence length
        
    Returns:
        sentiment: Predicted sentiment class
    """
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
    
    from src.data_processing import get_class_names
    class_names = get_class_names()
    return class_names[preds.item()]