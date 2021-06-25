import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('../Model Building/fitness.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/Prediction')
def prediction():
    return render_template('web.html')
@app.route('/Home')
def my_home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features =  [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['sad','neutral','happy','step_count','calories_burned','hours_of_sleep','weight_kg']

    df = pd.DataFrame(features_value, columns=features_name)

    output = model.predict(df)

    if(output[0] == 0):
        result = "You are not fit"
    else:
        result = "You are fit"

    return render_template('web.html',prediction_text=result)

if __name__ == '__main__':
    app.run(debug=False)
