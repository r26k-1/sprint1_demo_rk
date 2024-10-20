from transformers import pipeline
import pandas as pd
from transformers import pipeline

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def get_sentiment(text: str) -> str:
    """
    Perform sentiment analysis on the given text.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        str: The sentiment label (e.g., 'POSITIVE' or 'NEGATIVE').
    """
    if not text.strip():  # Check if text is empty
        return "No text provided for sentiment analysis."
    
    sentiment_result = sentiment_analyzer(text)  # Perform sentiment analysis
    return sentiment_result[0]['label']  # Return the sentiment label

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by filling and dropping NaN values."""
      # Fill NaN values with forward fill
    df.dropna(inplace=True)  # Drop any remaining NaN values
    return df

def get_sentiment(text: str) -> str:
    sentiment_pipeline = pipeline("sentiment-analysis")
    # Truncate text if it exceeds the max length
    text = text[:512]  # Truncate to the first 512 characters
    result = sentiment_pipeline(text)
    return result[0]

def generate_summary(text: str) -> str:
    summarization_pipeline = pipeline("summarization")
    # Truncate text if it exceeds the max length
    text = text[:1024]  # Truncate to the first 1024 characters
    summary = summarization_pipeline(text, max_length=300, min_length=50, do_sample=False)
    return summary[0]["summary_text"]
