"""
Flask API for Customer Churn Prediction.
Simple REST API with a single /predict endpoint.
"""

from flask import Flask, request, jsonify
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import predict_churn

app = Flask(__name__)


@app.route('/')
def home():
    """
    Home endpoint - API information.
    """
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Predict customer churn'
        }
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict churn for a customer.
    
    Expected JSON input:
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
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
        "MonthlyCharges": 70.35,
        "TotalCharges": 844.20
    }
    
    Returns:
    {
        "prediction": 0 or 1,
        "churn": "Yes" or "No"
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        # Make prediction
        prediction = predict_churn(data)
        
        # Return result
        result = {
            'prediction': prediction,
            'churn': 'Yes' if prediction == 1 else 'No'
        }
        
        return jsonify(result), 200
        
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n" + "="*50)
    print("Customer Churn Prediction API")
    print("="*50)
    print("\nStarting server on http://127.0.0.1:5000")
    print("\nEndpoints:")
    print("  GET  /         - API information")
    print("  POST /predict  - Predict customer churn")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)
