from flask import Flask, render_template, request
import pickle
import numpy as np


model = pickle.load(open('obesity_pickle.pkl', 'rb'))
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_obesity():
    age = float(request.form.get('age'))
    height = float(request.form.get('height'))
    weight = float(request.form.get('weight'))
    gender = str(request.form.get('gender'))
    smoking = str(request.form.get('smoking'))
    alcohol = str(request.form.get('alcohol'))
    family_overwt = request.form.get('family_overwt')

    result = model.predict(np.array([gender, age, height, weight, family_overwt, smoking, alcohol]).reshape(1, 7))
    if result[0] == 0:
        result = 'Insufficient Weight'
    if result[0] == 1:
        result = 'Normal Weight'
    if result[0] == 2:
        result = 'Type-I Obesity'
    if result[0] == 3:
        result = 'Type-II Obesity'
    if result[0] == 4:
        result = 'Type-III Obesity'
    if result[0] == 5:
        result = 'Level-I Overweight'
    if result[0] == 6:
        result = 'Level-II Overweight'

    return render_template('index.html', result = result)


if __name__ == '__main__':
    app.run(debug=True)
