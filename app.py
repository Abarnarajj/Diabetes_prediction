import joblib
from flask import Flask, request, jsonify
import pandas as pd

# Initialize Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('logistic_regression_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)

        # Convert incoming JSON data to a pandas DataFrame
        # Ensure the column order matches the training data (X_train)
        # The order of columns should be: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        # Example expected input: {"Pregnancies": 6, "Glucose": 148, ..., "Age": 50}
        input_df = pd.DataFrame([data])

        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Return prediction as JSON response
        return jsonify({
            'prediction': int(prediction[0]),
            'prediction_probability_class_0': float(prediction_proba[0][0]),
            'prediction_probability_class_1': float(prediction_proba[0][1])
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the Flask application
    # debug=True allows for automatic reloading on code changes and provides a debugger
    # host='0.0.0.0' makes the server externally visible to any host
    app.run(debug=True, host='0.0.0.0', port=5000)
