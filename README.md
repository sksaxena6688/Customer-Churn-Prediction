# Customer Churn Prediction

A simple, production-ready machine learning project to predict customer churn using Logistic Regression and Random Forest models.

## Problem Statement

Customer churn is when customers stop doing business with a company. Predicting churn helps businesses:
- Identify at-risk customers
- Take proactive retention actions
- Reduce revenue loss

This project builds a model to predict whether a customer will churn based on their usage patterns and demographics.

## Use Case

This system helps subscription-based businesses (telecom, SaaS, OTT, banking) 
identify customers who are likely to churn.

By predicting churn in advance, companies can:
- Target at-risk customers with retention offers
- Improve customer lifetime value (CLV)
- Reduce revenue loss
- Optimize marketing and support efforts

The trained model can be integrated into:
- CRM systems
- Customer support dashboards
- Automated retention campaigns


## Dataset
Note: The dataset used in this project is publicly available on Kaggle and included in this repository for ease of reproduction.

**Telco Customer Churn Dataset**

The dataset contains information about:
- Customer demographics (gender, senior citizen, partner, dependents)
- Services subscribed (phone, internet, streaming, etc.)
- Account information (tenure, contract type, payment method)
- Charges (monthly and total)
- **Target**: Churn (Yes/No)

**Download the dataset:**
1. Visit: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
2. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv`
3. Rename it to `churn.csv`
4. Place it in the `data/` folder

## Project Structure

```
customer-churn-prediction/
│
├── data/
│   └── churn.csv                 # Dataset (download required)
├── notebooks/
│   └── EDA.ipynb                 # Exploratory data analysis
├── src/
│   ├── preprocess.py             # Data preprocessing
│   ├── train.py                  # Model training
│   ├── predict.py                # Prediction logic
│   └── utils.py                  # Helper functions
├── model/
│   ├── churn_model.pkl           # Trained model (generated)
│   ├── scaler.pkl                # Scaler (generated)
│   ├── feature_names.pkl         # Feature names (generated)
│   └── numerical_indices.pkl     # Numerical indices (generated)
├── app.py                        # Flask API
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── .gitignore                    # Git ignore rules
```

## Installation

1. **Clone or download this project**

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the dataset** (see Dataset section above)

## Usage

### 1. Train the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train Logistic Regression (final model)
- Train Random Forest (comparison model)
- Print evaluation metrics
- Save the Logistic Regression model to `model/churn_model.pkl`


**Expected output:**
```
==================================================
TRAINING LOGISTIC REGRESSION (Baseline)
==================================================

Logistic Regression Performance:
Accuracy:  0.8045
Precision: 0.6721
Recall:    0.5543

==================================================
TRAINING RANDOM FOREST (Final Model)
==================================================

Random Forest Performance:
Accuracy:  0.7935
Precision: 0.6389
Recall:    0.4891
```

### 2. Run the Flask API

```bash
python app.py
```

The API will start on `http://127.0.0.1:5000`

### 3. Make Predictions

**Using curl:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "churn": "Yes"
}
```

**Using Python:**
```python
import requests

customer_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85
}

response = requests.post('http://127.0.0.1:5000/predict', json=customer_data)
print(response.json())
```

### 4. Explore the Data (Optional)

```bash
jupyter notebook notebooks/EDA.ipynb
```

## Model Details

### Models Trained
1. **Logistic Regression** – Final deployed model  
2. **Random Forest** – Comparative model for non-linear learning

### Features
- **Categorical**: Gender, Partner, Dependents, PhoneService, InternetService, Contract, PaymentMethod, etc.
- **Numerical**: Tenure, MonthlyCharges, TotalCharges

### Preprocessing
- Missing values in TotalCharges filled with median
- Categorical variables one-hot encoded
- Numerical features scaled using StandardScaler
- 80/20 train/test split

### Evaluation Metrics
- Accuracy
- Precision
- Recall

## API Endpoints

### GET /
Returns API information

### POST /predict
Predicts customer churn

**Request body:** JSON with customer features  
**Response:** `{"prediction": 0 or 1, "churn": "Yes" or "No"}`

## Future Improvements

1. **Model enhancements**
   - Try XGBoost or LightGBM
   - Hyperparameter tuning
   - Feature engineering

2. **API improvements**
   - Add authentication
   - Rate limiting
   - Batch predictions

3. **Deployment**
   - Containerize with Docker
   - Deploy to cloud (AWS, GCP, Azure)
   - Add monitoring and logging

4. **UI**
   - Build a simple web interface
   - Add visualization of predictions

## License

This project is for educational purposes.

## Author

Built following strict software engineering principles: simple, readable, production-ready code.
