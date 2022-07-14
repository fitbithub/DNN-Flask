import numpy as np
from tensorflow.keras.models import load_model

dnn_heart_pred = load_model('heart_disease_DNN.h5')


def deepneuralnetwork(int_features):
    array_feature = np.array(int_features)
    dimention_feature = np.expand_dims(array_feature, axis=0)
    prediction = dnn_heart_pred.predict(dimention_feature)
    output = (prediction > 0.5)
    if output == True:
        return " Presence of cardiovascular disease"
    else:
        return " Absence of cardiovascular disease"


