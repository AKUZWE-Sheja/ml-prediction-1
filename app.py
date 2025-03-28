import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

@app.before_request
def handle_options():
    if request.method == "OPTIONS":
        return _build_cors_preflight_response()
    
# Load model
with open('best_loan_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    model = artifacts['model']
    scaler = artifacts['scaler']

print("Model features:", model.feature_names_in_)
print("Scaler features:", artifacts.get('feature_names', 'Not available'))

def predict_loan(input_data):
    """Helper function for predictions"""
    num_cols = ['Principal', 'terms', 'age', 'past_due_days']
    input_data[num_cols] = scaler.transform(input_data[num_cols])
    proba = model.predict_proba(input_data)[0][1]
    return float(proba)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    education_map = artifacts['education_map']
    input_data = {
        'Principal': float(data['Principal']),
        'terms': float(data['terms']),
        'past_due_days': 0.0,
        'age': float(data['age']),
        'education': education_map[data['education']],
        'Gender': 0 if data['Gender'] == 'male' else 1
    }

    education_val = artifacts['education_map'].get(data.get('education'), None)
    if education_val is None:
        return jsonify({'error': 'Invalid or missing education field'}), 400

    
    feature_order = artifacts['features']
    input_df = pd.DataFrame([input_data])[feature_order]
    
    num_cols = ['Principal', 'terms', 'age', 'past_due_days']
    input_df[num_cols] = artifacts['scaler'].transform(input_df[num_cols])
    
    proba = artifacts['model'].predict_proba(input_df)[0][1]
    
    return jsonify({
        'prediction': 'Approved' if proba >= 0.5 else 'Rejected',
        'probability': float(proba)
    })


@app.route('/compute_graphs', methods=['POST', 'OPTIONS'])
def compute_graphs():
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        # Convert education using saved mapping
        education_val = artifacts['education_map'][data['education']]
        
        response = {}
        base_data = {
            'Principal': float(data['Principal']),
            'terms': float(data['terms']),
            'age': float(data['age']),
            'education': education_val,
            'Gender': 0 if data['Gender'].lower() == 'male' else 1,
            'past_due_days': 0
        }
        
        # Graph 1: Principal vs Approval Probability
        principals = np.linspace(500, 5000, 20)
        principal_data = []
        for p in principals:
            input_data = {**base_data, 'Principal': float(p)}
            input_df = pd.DataFrame([input_data])[artifacts['features']]
            input_df[artifacts['num_cols']] = artifacts['scaler'].transform(input_df[artifacts['num_cols']])
            proba = artifacts['model'].predict_proba(input_df)[0][1]
            principal_data.append(float(proba))
        
        response['principal_graph'] = {
            'x': principals.tolist(),
            'y': principal_data
        }
        
        # Graph 2: Age vs Approval Probability
        ages = np.linspace(18, 70, 20)
        age_data = []
        for age in ages:
            input_data = {**base_data, 'age': int(age)}
            input_df = pd.DataFrame([input_data])[artifacts['features']]
            input_df[artifacts['num_cols']] = artifacts['scaler'].transform(input_df[artifacts['num_cols']])
            proba = artifacts['model'].predict_proba(input_df)[0][1]
            age_data.append(float(proba))
        
        response['age_graph'] = {
            'x': ages.tolist(),
            'y': age_data
        }
        
        return _corsify_actual_response(jsonify(response))
        
    except Exception as e:
        return _corsify_actual_response(jsonify({'error': str(e)}), 500)
    
    # Graph 2: Age vs Approval Probability
    ages = np.linspace(18, 70, 20)
    response['age_graph'] = {
        'x': ages.tolist(),
        'y': [predict_loan(pd.DataFrame([{
            **data,
            'age': int(age),
            'past_due_days': 0
        }])) for age in ages]
    }
    
    return jsonify(response)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def _build_cors_preflight_response():
    response = jsonify({'message': 'CORS preflight'})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type")
    response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:5173")
    return response

if __name__ == '__main__':
    app.run(debug=True)