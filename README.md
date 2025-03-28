<!-- # Amazon Reviews Sentiment Analysis

A deep learning-based sentiment analysis system that classifies Amazon product reviews as positive, negative, or neutral. This project demonstrates the use of transformer models for natural language processing tasks.

## Features

- Uses the Hugging Face transformers library with DistilBERT
- Processes and analyzes Amazon product review text
- Classifies sentiment into three categories (Positive, Neutral, Negative)
- Includes visualization of model performance
- Provides an easy-to-use interface for analyzing new reviews

## Technologies

- Python 3.8+
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Pandas & NumPy
- Matplotlib & Seaborn

## Project Structure

```
amazon-review-sentiment-analysis/
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Functions for loading and processing review data
│   ├── model.py              # Model definition and training functions
│   └── evaluation.py         # Functions for model evaluation and visualization
│
├── scripts/
│   ├── train.py              # Script to train the model
│   └── evaluate.py           # Script to evaluate model performance
│
├── app.py                    # Simple demo application
├── main.py                   # Main entry point for running the pipeline
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore file
```


## Setup and Installation

1. Clone this repository: 
- git clone https://github.com/yourusername/amazon-review-sentiment-analysis.git
- cd amazon-review-sentiment-analysis

2. Create a virtual environment and install dependencies:
- python -m venv venv
- source venv/bin/activate  # On Windows: venv\Scripts\activate
- pip install -r requirements.txt

3. Run the demo application

python app.py

## Model Training and Evaluation

The model achieves **[XX]%** accuracy on the test set with the following performance metrics:

- Precision: **[XX]%**
- Recall: **[XX]%**
- F1 Score: **[XX]%**

![Confusion Matrix](link_to_confusion_matrix_image.png)

## Future Improvements

- Implement a more sophisticated labeling system beyond basic sentiment
- Add aspect-based sentiment analysis
- Deploy as a web service with API endpoints
- Explore different model architectures for comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Amazon Reviews Multi dataset from Hugging Face
- Hugging Face Transformers library -->






# Amazon Reviews Sentiment Analysis

A deep learning-based sentiment analysis system that classifies Amazon product reviews as positive, negative, or neutral. This project demonstrates the use of transformer models for natural language processing tasks.

## Features

* Uses the Hugging Face transformers library with DistilBERT
* Processes and analyzes Amazon product review text
* Classifies sentiment into three categories (Positive, Neutral, Negative)
* Includes visualization of model performance
* Provides an easy-to-use interface for analyzing new reviews

## Technologies

* Python 3.8+
* PyTorch
* Hugging Face Transformers
* scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
* Gradio (for demo application)

## Project Structure

```
amazon-review-sentiment-analysis/
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Functions for loading and processing review data
│   ├── model.py              # Model definition and training functions
│   └── evaluation.py         # Functions for model evaluation and visualization
│
├── scripts/
│   ├── train.py              # Script to train the model
│   └── evaluate.py           # Script to evaluate model performance
│
├── app.py                    # Simple demo application
├── main.py                   # Main entry point for running the pipeline
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
└── .gitignore                # Git ignore file
```

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/amazon-review-sentiment-analysis.git
   cd amazon-review-sentiment-analysis
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

To train the model using the default settings:

```
python main.py --train --use_huggingface
```

This will download the Amazon reviews dataset from Hugging Face and train the model.

Optional arguments:
- `--data_path PATH`: Use a local CSV file instead of the Hugging Face dataset
- `--sample_size N`: Use only N samples for faster experimentation  #50,000
- `--batch_size N`: Batch size for training (default: 16)
- `--max_length N`: Maximum sequence length (default: 128)
- `--learning_rate N`: Learning rate (default: 2e-5)
- `--epochs N`: Number of training epochs (default: 3)
- `--output_dir DIR`: Directory to save the model (default: models/sentiment)

### Evaluating the Model

To evaluate a trained model:

```
python main.py --evaluate --model_path models/sentiment
```

This will evaluate the model on the test set and generate evaluation metrics.

### Using the Demo Application

To run the interactive demo application:

```
python app.py
```

This will launch a Gradio web interface where you can enter reviews and get sentiment predictions.

## Model Performance

The model achieves the following performance metrics on the test set:

* Accuracy: 92.5%
* Precision: 91.8%
* Recall: 92.3%
* F1 Score: 92.0%

*Note: These are example metrics. Actual performance will depend on your specific training run.*

## Example Usage in Code

```python
from transformers import DistilBertTokenizer
from src.model import SentimentClassifier, predict_sentiment
import torch

# Load model and tokenizer
model_path = "models/sentiment"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentClassifier(n_classes=3)
model.load_state_dict(torch.load(f"{model_path}/best_model.pt", map_location=device))
model = model.to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Analyze a review
review = "This product is amazing! I love everything about it."
sentiment = predict_sentiment(review, model, tokenizer, device)
print(f"Sentiment: {sentiment}")
```

## Future Improvements

* Implement a more sophisticated labeling system beyond basic sentiment
* Add aspect-based sentiment analysis
* Deploy as a web service with API endpoints
* Explore different model architectures for comparison

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

* Amazon Reviews Multi dataset from Hugging Face
* Hugging Face Transformers library