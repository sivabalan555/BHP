from flask import Flask, request, jsonify, render_template
import pickle
import json
import numpy as np

# Load the model and columns
with open('./model/banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('./model/columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])

    # Prepare input data
    loc_index = data_columns.index(location.lower()) if location.lower() in data_columns else -1
    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    # Make prediction
    price = model.predict([x])[0]
    return render_template('home.html', prediction_text=f'Estimated House Price: ₹{price:.2f} Lakhs')

if __name__ == "__main__":
    app.run(debug=True)
