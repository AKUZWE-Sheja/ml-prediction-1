from flask import Flask, request, jsonify, render_template, make_response
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()

def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

def _corsify_actual_response(response, status=200):
    """Handle both response and status code with CORS headers"""
    if isinstance(response, tuple):
        response, status = response
    response = make_response(response, status)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Load model with all required artifacts
with open('best_loan_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    model = artifacts['model']
    scaler = artifacts['scaler']
    feature_order = artifacts['features']
    education_map = artifacts['education_map']
    num_cols = artifacts.get('num_cols', ['Principal', 'terms', 'age', 'past_due_days'])

print("Model features:", model.feature_names_in_)
print("Numerical columns:", num_cols)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        
        # Validate education value
        education_val = education_map.get(data.get('education'))
        if education_val is None:
            return _corsify_actual_response(jsonify({'error': 'Invalid education value'}), 400)

        # Prepare input data
        input_data = {
            'Principal': float(data['Principal']),
            'terms': float(data['terms']),
            'past_due_days': 0.0,
            'age': float(data['age']),
            'education': education_val,
            'Gender': 0 if data['Gender'].lower() == 'male' else 1
        }

        # Create DataFrame with correct feature order
        input_df = pd.DataFrame([input_data])[feature_order]
        input_df[num_cols] = scaler.transform(input_df[num_cols])
        
        # Make prediction
        proba = model.predict_proba(input_df)[0][1]
        
        return _corsify_actual_response(jsonify({
            'prediction': 'Approved' if proba >= 0.5 else 'Rejected',
            'probability': float(proba)
        }))
    except Exception as e:
        return _corsify_actual_response(jsonify({'error': str(e)}), 500)

@app.route('/feature_importance', methods=['POST', 'OPTIONS'])
def feature_importance():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        data = request.get_json()
        if not data:
            return _corsify_actual_response(jsonify({'error': 'No data provided'}), 400)
            
        # Validate education value
        education_val = education_map.get(data.get('education'))
        if education_val is None:
            return _corsify_actual_response(jsonify({'error': 'Invalid education value'}), 400)

        # Prepare base input data
        input_data = {
            'Principal': float(data['Principal']),
            'terms': float(data['terms']),
            'past_due_days': 0.0,
            'age': float(data['age']),
            'education': education_val,
            'Gender': 0 if data['Gender'].lower() == 'male' else 1
        }

        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models
            importances = np.abs(model.coef_[0])
        else:
            # Default to permutation importance if available or equal weights
            importances = np.ones(len(feature_order)) / len(feature_order)
        
        # Normalize importances to sum to 100%
        importances = 100 * (importances / importances.sum())
        
        # Create feature names mapping
        feature_names = {
            'Principal': 'Loan Amount',
            'terms': 'Loan Term',
            'age': 'Borrower Age',
            'education': 'Education Level',
            'Gender': 'Gender',
            'past_due_days': 'Past Due Days'
        }
        
        # Prepare response
        features = [feature_names.get(f, f) for f in feature_order]
        
        return _corsify_actual_response(jsonify({
            'features': features,
            'importances': importances.tolist()
        }))
        
    except Exception as e:
        return _corsify_actual_response(jsonify({'error': str(e)}), 500)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
