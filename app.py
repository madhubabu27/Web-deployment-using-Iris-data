import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('models/iris_classifier_model.pk', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    species = ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']
    with open('models/iris_classifier_model.pk', 'rb') as model_file: 
         model = pickle.load(model_file) 

    #species_class = int(model.predict([[petal_len, petal_wd, sepal_len, sepal_wd]])[0])
   
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    predict = model.predict(final_features)[0]

    output = species[predict]

    return render_template('index.html', prediction_text='Iris-family is {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)