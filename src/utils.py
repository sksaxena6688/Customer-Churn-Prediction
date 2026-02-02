"""
Utility functions for the Customer Churn Prediction project.
Simple helper functions for data loading and model evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def load_data(filepath):
    """
    Load the churn dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    df = pd.read_csv(filepath)
    return df


def evaluate_model(y_true, y_pred, model_name="Model"):
    """
    Print evaluation metrics for a model.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model for display
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    return accuracy, precision, recall


def print_data_info(df):
    """
    Print basic information about the dataset.
    
    Args:
        df: DataFrame to analyze
    """
    print("\nDataset Shape:", df.shape)
    print("\nColumn Names:")
    print(df.columns.tolist())
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
