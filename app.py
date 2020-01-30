
"""Web Application"""
"""Importing Libraries"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

"""Application is defined"""
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

"""Application root is defined"""
@app.route('/')
def home():
    return render_template('index.html')


"""Application route for prediction is defined"""
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)/200]
    prediction = model.predict(final_features)
    pred = prediction[0]
    output = pred*60
    if output >= 0:
        return render_template('index.html', prediction_text="Predicted Departure Delay in Minutes: {} \n Late".format(output))
    else:
        return render_template('index.html', prediction_text="Predicted Departure Delay in Minutes: {} \n Early".format(-10*output))


if __name__ == "__main__":
    app.run(debug=True)
