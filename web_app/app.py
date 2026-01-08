"""
Flask Web Application to display Gold Price Predictions.
Calls Lambda function via API Gateway to get predictions.
"""

import os
import requests
from flask import Flask, render_template, jsonify
from datetime import datetime

app = Flask(__name__)

# API Gateway URL - set as environment variable or update here
API_GATEWAY_URL = os.environ.get('API_GATEWAY_URL', 'https://n520h0gv1a.execute-api.eu-central-1.amazonaws.com/prod')
# For local testing without API Gateway, you can mock the response
MOCK_MODE = os.environ.get('MOCK_MODE', 'false').lower() == 'true'


def fetch_predictions_from_api():
    """
    Fetch predictions from API Gateway (which calls Lambda function).
    Returns (predictions_data, error_message) tuple.
    """
    # Mock mode for local testing
    if MOCK_MODE:
        return {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'generated_at': datetime.utcnow().isoformat(),
            'model_type': 'ARIMA',
            'model_name': 'arima_gold_price_production',
            'predictions': {
                'tomorrow': {
                    'price': 4234.97,
                    'horizon_days': 1
                },
                'day_after_tomorrow': {
                    'price': 4245.98,
                    'horizon_days': 2
                }
            },
            'note': 'Predictions generated daily. Model is retrained separately.'
        }, None
    
    if not API_GATEWAY_URL:
        return None, "API_GATEWAY_URL environment variable not set. Set MOCK_MODE=true for local testing."
    
    try:
        # Remove trailing slash if present
        url = API_GATEWAY_URL.rstrip('/')
        if not url.endswith('/predict'):
            url = f"{url}/predict"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # API Gateway returns JSON in body field
        data = response.json()
        
        # Parse API Gateway response format
        if 'body' in data:
            import json
            body_data = json.loads(data['body'])
            return body_data, None
        else:
            # Direct JSON response
            return data, None
            
    except requests.exceptions.RequestException as e:
        return None, f"Error fetching predictions: {str(e)}"
    except Exception as e:
        return None, f"Error parsing response: {str(e)}"


@app.route('/')
def index():
    """Main page - displays predictions."""
    predictions_data, error = fetch_predictions_from_api()
    
    return render_template('index.html',
                         predictions=predictions_data,
                         error=error)


@app.route('/api/predictions')
def api_predictions():
    """API endpoint - returns predictions as JSON."""
    predictions_data, error = fetch_predictions_from_api()
    
    if error:
        return jsonify({'success': False, 'error': error}), 500
    
    return jsonify(predictions_data)


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'gold-price-predictions-web'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))  # Lightsail verwendet 8080 oder 5000
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)