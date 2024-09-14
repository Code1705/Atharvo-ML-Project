from flask import Flask, request, render_template
import pandas as pd
from processing.pre_processing import pre_process_data
import pickle

app = Flask(__name__)

# Load pre-trained models
score_model_nb = pickle.load(open('naive_bayes_model.pkl', 'rb'))  # Naive Bayes Model
score_model_rf = pickle.load(open('random_forest_model.pkl', 'rb'))  # Random Forest Model

# Route for the home page
@app.route('/')
def home():
    return render_template('input.html')

# Route for uploading the CSV file and making predictions
@app.route('/upload', methods=['POST'])
def upload_csv():
    csv_file = request.files['file']  # Get the uploaded CSV file
    if csv_file:
        # Load CSV into a DataFrame
        data = pd.read_csv(csv_file)

        # Preprocess the data for prediction
        X_data = pre_process_data(data)

        # Make predictions using both models
        predictions_nb = score_model_nb.predict(X_data)  # Naive Bayes predictions
        predictions_rf = score_model_rf.predict(X_data)  # Random Forest predictions
        
        # Convert NumPy types to regular Python types for display
        predictions_nb = [int(pred) for pred in predictions_nb]
        predictions_rf = [int(pred) for pred in predictions_rf]
        
        # Assuming the original dataset has student IDs or names in the first column
        prediction_results = {
            f'{i+1}': {
                'Naive Bayes': ' ≥ 70' if pred_nb == 1 else ' < 70',
                'Random Forest': ' ≥ 70' if pred_rf == 1 else ' < 70'
            }
            for i, (pred_nb, pred_rf) in enumerate(zip(predictions_nb, predictions_rf))
        }

        # Render the input page again with the predictions displayed
        return render_template('input.html', prediction_results=prediction_results)

    return "No file uploaded", 400

if __name__ == '__main__':
    app.run(debug=True)
