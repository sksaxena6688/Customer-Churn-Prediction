"""
Test script for the Customer Churn Prediction API.
Demonstrates how to use the API with Python requests.
"""

import requests
import json

# API endpoint
API_URL = "http://127.0.0.1:5000/predict"

# Sample customer data - likely to churn
customer_high_risk = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": 85.50
}

# Sample customer data - likely to stay
customer_low_risk = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "Yes",
    "tenure": 60,
    "PhoneService": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "Yes",
    "OnlineBackup": "Yes",
    "DeviceProtection": "Yes",
    "TechSupport": "Yes",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Two year",
    "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 105.50,
    "TotalCharges": 6330.00
}


def test_prediction(customer_data, description):
    """
    Test the API with customer data.
    
    Args:
        customer_data: Dictionary with customer features
        description: Description of the test case
    """
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")
    
    try:
        response = requests.post(API_URL, json=customer_data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Prediction: {result['churn']}")
            print(f"  Value: {result['prediction']}")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  {response.json()}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Could not connect to API")
        print("  Make sure the API is running: python app.py")
    except Exception as e:
        print(f"✗ Error: {str(e)}")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Customer Churn Prediction API - Test Script")
    print("="*60)
    
    # Test high-risk customer
    test_prediction(customer_high_risk, "High-Risk Customer (new, month-to-month)")
    
    # Test low-risk customer
    test_prediction(customer_low_risk, "Low-Risk Customer (long tenure, 2-year contract)")
    
    print("\n" + "="*60)
    print("Testing complete!")
    print("="*60 + "\n")
