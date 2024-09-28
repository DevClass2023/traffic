from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scalers
model = joblib.load('bugmodel.pkl')
x_scaler = joblib.load('bugscalers.pkl')
label_encoder = joblib.load('bugencoders.pkl')

# Home page route
@app.route('/')
def home():
    return render_template('homess.html')

# Prediction page route
@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# Dashboard page route
@app.route('/dashboards')
def dashboard():
    return render_template('dashboards.html')

# Prediction result route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Gather input data from form
        d = {
            'is_holiday': int(request.form.get('isholiday') == 'yes'),
            'temperature': float(request.form.get('temperature')),
            'day': int(request.form.get('day', 0)),
            'hour': int(request.form.get('time', '00')[:2]),
            'month_day': int(request.form.get('date')[8:]),
            'year': int(request.form.get('date')[:4]),
            'month': int(request.form.get('date')[5:7]),
        }

        # Prepare features for the model
        features = [d['is_holiday'], d['temperature'], d['day'], d['hour'], d['month_day'], d['year'], d['month']]
        scaled_features = x_scaler.transform([features])

        # Make a prediction
        prediction = model.predict(scaled_features)
        traffic_category = label_encoder.inverse_transform(prediction)[0]

        # Suggestions based on traffic category
        suggestions = {
            'No Traffic': 'No traffic currently. You can travel without any delay.',
            'Low Traffic': 'Low traffic. You can travel with minimal delay.',
            'Medium Traffic': 'Medium traffic. Consider traveling at a different time if possible.',
            'High Traffic': 'High traffic. Avoid traveling during peak hours or consider alternative routes.'
        }
        suggestion = suggestions.get(traffic_category, 'No suggestion available.')

        # Render the result page with the prediction and suggestion
        return render_template('result.html', traffic_category=traffic_category, suggestion=suggestion)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
