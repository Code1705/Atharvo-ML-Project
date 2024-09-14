from flask import Flask, request, jsonify,render_template
import pandas as pd
from processing.pre_processing import pre_process_data
import pickle

app = Flask(__name__)
score_model1 = pickle.load(open('naive_bayes_model.pkl', 'rb'))
score_model = pickle.load(open('random_forest_model.pkl', 'rb'))
#Defines a route for the home page
@app.route('/')
def home():
    return render_template('input.html')

# Route for uploading CSV file
@app.route('/upload', methods=['POST'])
def upload_csv():    
    csv_file = request.files['file']
    if csv_file:
        data = pd.read_csv(csv_file)
        X_data = pre_process_data(data)            
        response_final_nb=placement_model.predict(X_data)
        response_final_rf=placement_model1.predict(X_data)
        p_status_lr='Placed' if response_final_lr[0]==1 else 'Not Placed'
        p_status_rfc='Placed' if response_final_rfc[0]==1 else 'Not Placed'
        return render_template('input.html',prediction_text_lr='Placement Status = {}'.format(p_status_lr),prediction_text_rfc='Placement Status = {}'.format(p_status_rfc))        
    # else:
    #     return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(debug=True)
