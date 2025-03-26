import os
import sys
from src.logger import logging
from src.exception import Customexception
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_transformation import load_data, feature_engineering, split_data



def train_model(X_train, y_train, model_type="GradientBoosting"):
    """Trains a specified model type."""
    try:
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=10, min_samples_split=4, min_samples_leaf=2, random_state=42)
        elif model_type == "naive_bayes":
            model = MultinomialNB()
        else:
            logging.error("Invalid model type specified.")
            return None
        
        model.fit(X_train, y_train)
        logging.info(f"{model_type} model trained successfully.")
        return model
    except Exception as e:
        logging.error(f"Error in model training")
        return Customexception(e,sys)

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and logs performance metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        matrix = confusion_matrix(y_test, y_pred)
        
        logging.info(f"Model Accuracy: {accuracy}")
        logging.info(f"Classification Report:\n{report}")
        logging.info(f"Confusion Matrix:\n{matrix}")
        
        return accuracy, report, matrix
    except Exception as e:
        logging.error(f"Error in model evaluation")
        return Customexception(e,sys)

# Example usage
"""if __name__ == "__main__":
    file_path = "your_dataset.csv"  # Update with actual path
    target_column = "your_target_column"  # Update with actual target column
    
    df = load_data(file_path)
    if df is not None:
        df = feature_engineering(df)
        if df is not None:
            X_train, X_test, y_train, y_test = split_data(df, target_column)
            model = train_model(X_train, y_train, model_type="random_forest")
            if model:
                evaluate_model(model, X_test, y_test)
"""