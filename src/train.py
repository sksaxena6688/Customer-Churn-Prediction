"""
Model training module for Customer Churn Prediction.
Trains Logistic Regression and Random Forest models.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import evaluate_model
from src.preprocess import preprocess_data


def train_models(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression and Random Forest models.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        best_model: The Random Forest model (chosen as final model)
    """
    
    print("\n" + "="*50)
    print("TRAINING LOGISTIC REGRESSION (Baseline)")
    print("="*50)
    
    # Train Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    # Evaluate Logistic Regression
    lr_pred = lr_model.predict(X_test)
    evaluate_model(y_test, lr_pred, "Logistic Regression")
    
    print("\n" + "="*50)
    print("TRAINING RANDOM FOREST (Final Model)")
    print("="*50)
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest
    rf_pred = rf_model.predict(X_test)
    evaluate_model(y_test, rf_pred, "Random Forest")
    
    # Random Forest is chosen as the final model
    print("\n" + "="*50)
    print("FINAL MODEL: Random Forest")
    print("="*50)
    
    return rf_model


def save_model(model, filepath='model/churn_model.pkl'):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model to save
        filepath: Path where model will be saved
    """
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")


if __name__ == "__main__":
    print("Starting model training pipeline...")
    
    # Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data('data/churn.csv')
    
    # Train models
    final_model = train_models(X_train, X_test, y_train, y_test)
    
    # Save the final model
    save_model(final_model)
    
    print("\n✓ Training complete!")
