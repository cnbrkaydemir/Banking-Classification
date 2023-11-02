import pandas as pd
import numpy as np
import pickle


def predict(customer_data):
    model = pickle.load(open('lgb_model.sav', 'rb'))

    arr = np.asarray(customer_data)
    
    customer_data = arr.reshape(1,-1)


    prediction = model.predict(customer_data)


    return prediction
