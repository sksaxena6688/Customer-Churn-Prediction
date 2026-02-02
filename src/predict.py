"""
Prediction module for Customer Churn Prediction.
Loads trained model and makes predictions on new data.
"""

import pandas as pd
import numpy as np
import joblib
import os


def load_model(model_path='model/churn_model.pkl'):
    """
    Load the trained model from disk.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    model = joblib.load(model_path)
    return model


def load_preprocessors():
    """
    Load scaler and feature names used during training.
    
    Returns:
        scaler, feature_names, numerical_indices
    """
    scaler = joblib.load('model/scaler.pkl')
    feature_names = joblib.load('model/feature_names.pkl')
    numerical_indices = joblib.load('model/numerical_indices.pkl')
    
    return scaler, feature_names, numerical_indices


def prepare_input(input_dict, feature_names, scaler, numerical_indices):
    """
    Convert input dictionary to model-ready format.
    
    Args:
        input_dict: Dictionary with customer data
        feature_names: List of feature names from training
        scaler: Fitted StandardScaler
        numerical_indices: Indices of numerical columns
        
    Returns:
        Numpy array ready for prediction
    """
    # Create a dataframe from input
    input_df = pd.DataFrame([input_dict])
    
    # Get categorical columns (those that are objects)
    categorical_cols = input_df.select_dtypes(include=['object']).columns.tolist()
    
    # One-hot encode (same as training)
    input_encoded = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)
    
    # Ensure all features from training are present
    for feature in feature_names:
        if feature not in input_encoded.columns:
            input_encoded[feature] = 0
    
    # Reorder columns to match training
    input_encoded = input_encoded[feature_names]
    
    # Scale numerical features
    input_scaled = input_encoded.copy()
    input_scaled.iloc[:, numerical_indices] = scaler.transform(input_encoded.iloc[:, numerical_indices])
    
    return input_scaled.values


def predict_churn(input_dict):
    """
    Predict churn for a single customer.
    
    Args:
        input_dict: Dictionary containing customer features
        
    Returns:
        prediction: 0 (No Churn) or 1 (Churn)
    """
    # Load model and preprocessors
    model = load_model()
    scaler, feature_names, numerical_indices = load_preprocessors()
    
    # Prepare input
    X = prepare_input(input_dict, feature_names, scaler, numerical_indices)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return int(prediction)


if __name__ == "__main__":
    # Test prediction with sample data
    sample_customer = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 70.35,
        'TotalCharges': 844.20
    }
    
    try:
        prediction = predict_churn(sample_customer)
        result = "CHURN" if prediction == 1 else "NO CHURN"
        print(f"\nPrediction: {result} (value: {prediction})")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run 'python src/train.py' first to train the model.")
