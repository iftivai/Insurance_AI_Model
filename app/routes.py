from flask import render_template, request, jsonify
from .models import model, train_model
import numpy as np

def home():
    return render_template('index.html')

def predict():
    data = request.form
    input_data = np.array([
        float(data['claim_amount']),
        int(data['num_previous_claims']),
        int(data['customer_age']),
        int(data['policy_age']),
        int(data['incident_severity']),
        int(data['witness_present'])
    ]).reshape(1, -1)
    prediction = model.predict(input_data)
    return render_template('index.html', prediction=f'Fraudulent: {int(prediction[0])}')

def retrain():
    file = request.files['file']
    if file:
        df = train_model(file)
        return jsonify({'message': 'Model retrained successfully'})
    return jsonify({'error': 'File upload failed'})
