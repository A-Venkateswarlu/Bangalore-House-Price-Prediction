from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("cleaned_data.csv")
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bedrooms')
    bath = request.form.get('bathrooms')
    sqft = request.form.get('size')

    # Debug print statements
    print("Location:", location)
    print("BHK:", bhk)
    print("Bath:", bath)
    print("Sqft:", sqft)

    if None in (location, bhk, bath, sqft):
        return "Missing data", 400

    try:
        input_df = pd.DataFrame([[location, float(sqft), float(bath), float(bhk)]], columns=['location', 'total_sqft', 'bath', 'bhk'])
        prediction = pipe.predict(input_df)[0] * 1e5
        return str(np.round(prediction, 2))
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(debug=True)
