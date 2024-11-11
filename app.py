from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and scaler
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please run model.py to generate these files.")
    model = None
    scaler = None



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/fraud-detection', methods=['GET', 'POST'])
def fraud_detection():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # Get the single line input and split by comma
            feature_input = request.form.get('features')
            transaction_data = [float(x.strip()) for x in feature_input.split(',')]
            
            # Validate that there are exactly 30 features
            if len(transaction_data) != 30:
                prediction_text = "Error: Please enter exactly 30 feature values."
            elif model and scaler:
                # Scale the input data
                transaction_data = scaler.transform([transaction_data[:-1]])  # Exclude 'Class' for prediction
                
                # Predict
                prediction = model.predict(transaction_data)
                prediction_text = "Fraudulent" if prediction[0] == 1 else "Legitimate"
            else:
                prediction_text = "Model not loaded. Please check if the model and scaler files exist."
        except ValueError:
            prediction_text = "Error: Please ensure all inputs are valid numbers separated by commas."
        except Exception as e:
            print(f"Error during prediction: {e}")
            prediction_text = "An error occurred during prediction. Please check your input and try again."
    
    return render_template('fraud_detection.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
