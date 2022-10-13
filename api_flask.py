# -*- coding: utf-8 -*-

import dill as pickle
import json
from flask import Flask, jsonify, request
from functions import predict
import numpy as np

app = Flask(__name__)
best_model = pickle.load(open('LR_clf.sav', 'rb')) 
f = open('LR_params.json')
thres_dict = json.load(f)
thres = thres_dict['Threshold']
scaler = pickle.load(open('MinMaxScaler_LR.sav', 'rb'))

@app.route('/')
def index():
    return 'Projet 7 - Implémentez un modèle de scoring'

@app.route("/predict", methods=['POST'])
def get_prediction():
    ind = request.form.to_dict()
    ind = list(ind.values())
    ind = list(map(float, ind))
    ind = np.array(ind).reshape(1, -1)
    rep, proba = predict(ind, best_model, scaler, thres)
    d = {
        'rep' : rep,
        'proba' : proba
        }
 
    return (d)
 
if __name__ == '__main__':
    
     app.run(debug=True, host='0.0.0.0')
    # app.run()