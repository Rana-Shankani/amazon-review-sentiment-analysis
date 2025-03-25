# Amazon Reviews Sentiment Analysis

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
├── notebooks/
│   └── model_development.ipynb    # Exploratory analysis and model development
│
├── src/
│   ├── init.py
│   ├── data_processing.py         # Functions for loading and processing review data
│   ├── model.py                   # Model definition and training functions
│   └── evaluation.py              # Functions for model evaluation and visualization
│
├── app.py                         # Simple demo application
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── .gitignore                     # Git ignore file
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
- Hugging Face Transformers library
