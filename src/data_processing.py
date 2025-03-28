import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

class AmazonReviewDataset(Dataset):
    def __init__(self, reviews, targets, tokenizer, max_length=128):
        """
        Custom dataset for Amazon reviews
        
        Args:
            reviews: List of review texts
            targets: List of sentiment labels (0: negative, 1: neutral, 2: positive)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.reviews)
    
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }


def load_data(file_path=None, use_huggingface=True, sample_size=None):
    """
    Load Amazon review data either from a local file or HuggingFace datasets
    
    Args:
        file_path: Path to local CSV file (optional)
        use_huggingface: Whether to use HuggingFace dataset
        sample_size: Number of samples to use (for faster experimentation)
        
    Returns:
        train_df, test_df: DataFrames containing training and test data
    """
    if use_huggingface:
        from datasets import load_dataset
        dataset = load_dataset("mteb/amazon_reviews_multi", trust_remote_code=True)
        
        # Convert to pandas for easier manipulation
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        # Map stars to sentiment categories (1-2: negative, 3: neutral, 4-5: positive)
        train_df['sentiment'] = train_df['label'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
        test_df['sentiment'] = test_df['label'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
        
        # Limit sample size if specified
        if sample_size:
            train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(min(sample_size // 5, len(test_df)), random_state=42)
            
        return train_df, test_df
    else:
        if file_path is None:
            raise ValueError("File path must be provided when use_huggingface is False")
        
        df = pd.read_csv(file_path)
        
        # Assuming the file has 'review_text' and 'rating' columns
        # Map ratings to sentiment categories (1-2: negative, 3: neutral, 4-5: positive)
        df['sentiment'] = df['rating'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else 2))
        
        # Split into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        
        # Limit sample size if specified
        if sample_size:
            train_df = train_df.sample(min(sample_size, len(train_df)), random_state=42)
            test_df = test_df.sample(min(sample_size // 5, len(test_df)), random_state=42)
            
        return train_df, test_df

def create_data_loaders(train_df, test_df, tokenizer, batch_size=16, max_length=128):
    """
    Create PyTorch DataLoaders for training and testing
    
    Args:
        train_df: Training DataFrame
        test_df: Testing DataFrame
        tokenizer: DistilBERT tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing
    """
    # Split train into train and validation
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    train_dataset = AmazonReviewDataset(
        reviews=train_df['text'].to_numpy(),
        targets=train_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = AmazonReviewDataset(
        reviews=val_df['text'].to_numpy(),
        targets=val_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = AmazonReviewDataset(
        reviews=test_df['text'].to_numpy(),
        targets=test_df['sentiment'].to_numpy(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader, test_loader

def get_class_names():
    """Return the class names for sentiment labels"""
    return ["Negative", "Neutral", "Positive"]