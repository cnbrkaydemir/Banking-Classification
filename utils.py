import pandas as pd
import numpy as np
import tensorflow as tf
import pickle


def predict(customer_data):
    model = tf.keras.models.load_model('models/ann_model.keras')

    scaler = pickle.load(open('models/scaler.sav', 'rb'))


    arr = np.asarray(customer_data)


    customer_data = scaler.fit_transform(arr.reshape(1,-1))



    prediction = model.predict(customer_data)


    return prediction
