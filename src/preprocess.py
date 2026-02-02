"""
Data preprocessing module for Customer Churn Prediction.
Handles data cleaning, encoding, scaling, and train/test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_data(filepath, test_size=0.2, random_state=42):
    """
    Load and preprocess the churn dataset.
    
    Steps:
    1. Load data
    2. Drop customerID
    3. Handle missing values in TotalCharges
    4. Separate features and target
    5. One-hot encode categorical variables
    6. Scale numerical features
    7. Split into train/test sets
    
    Args:
        filepath: Path to the CSV file
        test_size: Proportion of data for testing (default 0.2)
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test as numpy arrays
    """
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    
    # Drop customerID (not useful for prediction)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Handle missing values in TotalCharges
    # Convert TotalCharges to numeric (some values might be strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill missing TotalCharges with median
    if df['TotalCharges'].isnull().sum() > 0:
        print(f"Filling {df['TotalCharges'].isnull().sum()} missing TotalCharges values...")
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Separate target variable
    # Convert Churn to binary (Yes=1, No=0)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop('Churn', axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"\nCategorical columns: {len(categorical_cols)}")
    print(f"Numerical columns: {len(numerical_cols)}")
    
    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print(f"\nShape after encoding: {X_encoded.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Scale numerical features
    scaler = StandardScaler()
    
    # Get indices of numerical columns in the encoded dataframe
    numerical_indices = [i for i, col in enumerate(X_encoded.columns) if col in numerical_cols]
    
    # Scale only numerical columns
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled.iloc[:, numerical_indices] = scaler.fit_transform(X_train.iloc[:, numerical_indices])
    X_test_scaled.iloc[:, numerical_indices] = scaler.transform(X_test.iloc[:, numerical_indices])
    
    # Save scaler and column names for later use
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(X_encoded.columns.tolist(), 'model/feature_names.pkl')
    joblib.dump(numerical_indices, 'model/numerical_indices.pkl')
    
    print("\nPreprocessing complete!")
    
    # Convert to numpy arrays
    return (
        X_train_scaled.values,
        X_test_scaled.values,
        y_train.values,
        y_test.values
    )


if __name__ == "__main__":
    # Test the preprocessing
    X_train, X_test, y_train, y_test = preprocess_data('data/churn.csv')
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
