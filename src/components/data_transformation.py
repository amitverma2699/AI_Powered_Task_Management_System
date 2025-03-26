import sys
import os
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import Customexception
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re

# Configure logging
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def load_data(file_path):
    """Loads dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully with shape {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return Customexception(e,sys)
def preprocess_text(text):
    """Cleans and preprocesses text data."""
    try:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        words = word_tokenize(text)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error in text preprocessing")
        return Customexception(e,sys)

def feature_engineering(df):
    """Applies feature transformation like label encoding, scaling, and TF-IDF."""
    try:
        # Handling categorical variables
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        
        # Scaling numerical data
        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        df[num_cols] = scaler.fit_transform(df[num_cols])
        
        logging.info("Feature engineering applied successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in feature engineering")
        return Customexception(e,sys)

def split_data(df, target_column):
    """Splits data into training and testing sets."""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info("Data split into train and test sets.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data splitting")
        return Customexception(e,sys)

# Example usage
"""if __name__ == "__main__":
    file_path = "your_dataset.csv"  # Update with actual path
    df = load_data(file_path)
    
    if df is not None:
        df = feature_engineering(df)
        if df is not None:
            X_train, X_test, y_train, y_test = split_data(df, target_column='your_target_column')
"""