import gradio as gr
import torch
import os
from transformers import DistilBertTokenizer
from src.model import SentimentClassifier, predict_sentiment

def load_sentiment_model(model_path="models/sentiment"):
    """
    Load the sentiment analysis model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        model, tokenizer, device: Loaded model components
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = SentimentClassifier(n_classes=3)
        model.load_state_dict(torch.load(f"{model_path}/best_model.pt", map_location=device))
        model = model.to(device)
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, device

def analyze_review(review_text, model=None, tokenizer=None, device=None):
    """
    Analyze a product review for sentiment
    
    Args:
        review_text: Text of the review
        model, tokenizer, device: Model components
        
    Returns:
        Analysis result as string
    """
    if model is None or tokenizer is None:
        return "Model not loaded. Please train the model first."
    
    if not review_text.strip():
        return "Please enter a review to analyze."
    
    sentiment = predict_sentiment(review_text, model, tokenizer, device)
    confidence = get_confidence(review_text, model, tokenizer, device)
    
    # Format the result
    result = f"Sentiment: {sentiment}\nConfidence: {confidence:.2f}%\n\n"
    
    # Add interpretation
    if sentiment == "Positive":
        result += "This review expresses favorable opinions about the product."
    elif sentiment == "Negative":
        result += "This review expresses unfavorable opinions about the product."
    else:
        result += "This review expresses mixed or neutral opinions about the product."
    
    return result

def get_confidence(text, model, tokenizer, device, max_length=128):
    """
    Get confidence score for prediction
    
    Args:
        text: Review text
        model, tokenizer, device: Model components
        max_length: Maximum sequence length
        
    Returns:
        Confidence percentage
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
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, _ = torch.max(probabilities, dim=1)
    
    return confidence.item() * 100

def create_demo():
    """Create and launch the Gradio demo"""
    
    # Try to load the model
    model, tokenizer, device = load_sentiment_model()
    model_status = "Model loaded successfully" if model is not None else "Model not loaded. Please train the model first."
    
    # Define sample reviews
    sample_reviews = [
        "This product is amazing! It works exactly as described and the quality is outstanding.",
        "Terrible experience. The item arrived damaged and customer service was unhelpful.",
        "The product is okay. Nothing special but it gets the job done.",
        "Not sure how I feel about this purchase. It has some good features but also a few drawbacks."
    ]
    
    # Create the Gradio interface
    demo = gr.Interface(
        fn=lambda text: analyze_review(text, model, tokenizer, device),
        inputs=gr.Textbox(lines=5, placeholder="Enter an Amazon product review here..."),
        outputs=gr.Textbox(label="Analysis Result"),
        title="Amazon Review Sentiment Analysis",
        description=f"Enter a product review to analyze its sentiment (positive, neutral, or negative). {model_status}",
        examples=sample_reviews
    )
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()