import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from model.model_utils import preprocess_input
from model.database import HistoryDatabase
import pandas as pd
import json

app = Flask(__name__)

# Add error handling and logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Load model
    # MODEL_PATH = os.path.join('model', 'rf_model.joblib')
    # model = joblib.load(MODEL_PATH)
    model = joblib.load('model/random_forest_model2.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Create database instance
db = HistoryDatabase()

# Load encoders and scalers if they exist, otherwise create new ones
if os.path.exists('model/encoders.joblib'):
    encoders = joblib.load('model/encoders.joblib')
else:
    encoders = {}

if os.path.exists('model/scaler.joblib'):
    scaler = joblib.load('model/scaler.joblib')
else:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

# Define feature information
feature_info = {
    'ProviderClaimCount': list(range(1, 10001)),
    'AttendingPhysician': list(range(0, 707)),
    'OtherPhysician': list(range(0, 163)),
    'State': list(range(1, 55)),
    'OperatingPhysician': list(range(0, 56)),
    'County': list(range(1, 1000)),
    'DiagnosisGroupCode': list(range(0, 118)),
    'phy_same': [0, 1, 2, 3],
    'phy_count': [0, 1, 2, 3]
}

# Feature importance descriptions
feature_importance = {
    'ProviderClaimCount': 'Abnormal provider claim count may indicate systematic fraud',
    'AttendingPhysician': 'Claim patterns of attending physician may show improper behavior',
    'OtherPhysician': 'Multiple physicians involved may indicate excessive medical services',
    'State': 'Certain regions may have higher fraud risk',
    'OperatingPhysician': 'Operation frequency of surgeon may show abnormal patterns',
    'County': 'Specific counties may have higher fraud risk',
    'DiagnosisGroupCode': 'Mismatch between diagnosis code and treatment may indicate fraud',
    'DeductibleAmtPaid': 'Abnormal deductible payment patterns may indicate issues',
    'phy_same': 'Physician role overlap may indicate improper behavior',
    'phy_count': 'Abnormal number of physicians may indicate excessive medical services'
}

@app.route('/')
@app.route('/home')
def index():
    """Render homepage"""
    return render_template('index.html', feature_info=feature_info)

@app.route('/autocomplete/<feature>', methods=['GET'])
def autocomplete(feature):
    term = request.args.get('term', '')
    if feature in feature_info:
        suggestions = [str(item) for item in feature_info[feature] 
                      if str(item).lower().startswith(term.lower())]
        return jsonify(suggestions[:10])  # Limit to 10 suggestions
    return jsonify([])

@app.route('/validate/<feature>', methods=['POST'])
def validate(feature):
    value = request.json.get('value')
    if feature in feature_info:
        try:
            value = int(value) if feature != 'DeductibleAmtPaid' else float(value)
            valid = value in feature_info[feature]
            if not valid and feature in ['ProviderClaimCount', 'AttendingPhysician', 'OtherPhysician', 
                                       'State', 'OperatingPhysician', 'County', 'DiagnosisGroupCode']:
                # For integer range features, check if value is within range
                min_val = min(feature_info[feature])
                max_val = max(feature_info[feature])
                valid = min_val <= value <= max_val
            return jsonify({'valid': valid})
        except (ValueError, TypeError):
            return jsonify({'valid': False})
    return jsonify({'valid': True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data and convert
        data = {
            'ProviderClaimCount': int(request.form['provider_claim_count']),
            'DeductibleAmtPaid': float(request.form['deductible_amt_paid']),
            'AttendingPhysician': int(request.form['attending_physician']),
            'OperatingPhysician': int(request.form['operating_physician']),
            'OtherPhysician': int(request.form['other_physician']),
            'State': int(request.form['state']),
            'County': int(request.form['county']),
            'DiagnosisGroupCode': int(request.form['diagnosis_group_code']),
            'phy_same': int(request.form['phy_same']),
            'phy_count': int(request.form['phy_count'])
        }

        # Print converted data types for debugging
        for key, value in data.items():
            logger.info(f"{key}: {value} (type: {type(value).__name__})")

        # Validate input values
        for feature, value in data.items():
            if feature in feature_info:
                if feature in ['ProviderClaimCount', 'AttendingPhysician', 'OtherPhysician', 
                            'State', 'OperatingPhysician', 'County', 'DiagnosisGroupCode']:
                    # For integer range features, check if value is within range
                    min_val = min(feature_info[feature])
                    max_val = max(feature_info[feature])
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"Invalid value for {feature}: {value}, should be between {min_val} and {max_val}")
                elif feature in ['phy_same', 'phy_count'] and value not in feature_info[feature]:
                    raise ValueError(f"Invalid value for {feature}: {value}, should be 0, 1, 2, or 3")

        # Create feature DataFrame
        features = pd.DataFrame([data])
        
        # Process features (no encoding needed as all features are numeric)
        # Ensure correct feature order
        features = features[['ProviderClaimCount', 'AttendingPhysician', 'OtherPhysician', 
                             'State', 'OperatingPhysician', 'County', 'DiagnosisGroupCode', 
                             'DeductibleAmtPaid', 'phy_same', 'phy_count']]
        
        # Make prediction
        prediction = model.predict_proba(features)[0]
        fraud_probability = prediction[1]  # Assuming 1 represents fraud class
        
        # Determine risk level based on probability
        if fraud_probability < 0.3:
            risk_level = "Low Risk"
            risk_class = "success"
        elif fraud_probability < 0.7:
            risk_level = "Medium Risk"
            risk_class = "warning"
        else:
            risk_level = "High Risk"
            risk_class = "danger"
        
        # Save to history
        db.add_record(data, fraud_probability, risk_level)
        
        return render_template(
            'result.html',
            prediction=round(fraud_probability * 100, 2),
            risk_level=risk_level,
            risk_class=risk_class,
            data=data,
            feature_importance=feature_importance
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', error=str(e), feature_info=feature_info)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint, returns prediction results in JSON format"""
    if request.method == 'POST':
        try:
            data = request.get_json()
            processed_data = preprocess_input(data)
            
            prediction = model.predict_proba([processed_data])[0]
            fraud_probability = prediction[1]
            
            return jsonify({
                'fraud_probability': float(fraud_probability),
                'risk_level': get_risk_level(fraud_probability)
            })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': 'Invalid request method'}), 405

def get_risk_level(probability):
    """Determine risk level based on probability"""
    if probability < 0.3:
        return 'Low Risk'
    elif probability < 0.7:
        return 'Medium Risk'
    else:
        return 'High Risk'

def get_risk_class(probability):
    """Determine CSS class based on probability"""
    if probability < 0.3:
        return 'success'
    elif probability < 0.7:
        return 'warning'
    else:
        return 'danger'

@app.route('/history')
def history():
    """Display history page"""
    records = db.get_all_records()
    return render_template('history.html', history=records)

@app.route('/record/<int:record_id>')
def view_record(record_id):
    """View details of a specific record"""
    record = db.get_record_by_id(record_id)
    if record:
        # Check if it's an old record (using old field names)
        form_data = record['form_data']
        
        # If it's an old format record, convert to new format
        if 'policy_annual_premium' in form_data:
            # Create new format data
            new_form_data = {
                'ProviderClaimCount': int(form_data.get('policy_annual_premium', 0)),
                'DeductibleAmtPaid': float(form_data.get('umbrella_limit', 0)),
                'AttendingPhysician': 1,  # Default value
                'OperatingPhysician': 1,  # Default value
                'OtherPhysician': 1,      # Default value
                'State': 1,               # Default value
                'County': 1,              # Default value
                'DiagnosisGroupCode': 1,  # Default value
                'phy_same': 0,            # Default value
                'phy_count': 1            # Default value
            }
            # Update record
            form_data = new_form_data
        
        return render_template(
            'result.html',
            prediction=record['prediction'] * 100,  # Convert to percentage
            risk_level=record['risk_level'],
            risk_class=get_risk_class(record['prediction']),
            data=form_data,
            feature_importance=feature_importance,
            from_history=True
        )
    return redirect(url_for('history'))

@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    """Delete a record from history"""
    if db.delete_record(record_id):
        # Return a success response for AJAX calls
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': True, 'message': 'Record deleted successfully'})
        # For regular form submissions, redirect back to history page
        return redirect(url_for('history'))
    else:
        # Return an error response for AJAX calls
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'message': 'Record not found'}), 404
        # For regular form submissions, redirect back to history page with an error
        return redirect(url_for('history'))

@app.route('/dashboard')
def dashboard():
    """Display data analysis dashboard"""
    history = db.get_all_records()
    
    # Calculate statistics
    stats = {
        'total_cases': len(history),
        'high_risk_cases': len([x for x in history if x['prediction'] > 0.7]),
        'avg_fraud_rate': sum(x['prediction'] for x in history) / len(history) if history else 0,
        'most_common_state': max(set(x['form_data']['State'] for x in history), key=lambda x: list(r['form_data']['State'] for r in history).count(x)) if history else 'N/A'
    }
    
    return render_template('dashboard.html', stats=stats, history=history)

@app.route('/export_report/<int:record_id>')
def export_report(record_id):
    """Export PDF report"""
    record = db.get_record_by_id(record_id)
    # Generate PDF report
    return send_file('report.pdf', as_attachment=True)

@app.errorhandler(403)
def forbidden_error(error):
    return render_template('index.html', error="Access forbidden. Please try again."), 403

@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Page not found. Please try again."), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error. Please try again."), 500

if __name__ == '__main__':
    if model is None:
        logger.error("Application starting with no model loaded!")
    # 修改为支持云部署的代码
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 