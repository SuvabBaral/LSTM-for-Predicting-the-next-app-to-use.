import os
import keras
from keras.models import model_from_json

#load the pre-trained RNN model

def getTrainedModel(modelFilePath = './../Models/trainedModel.json', weightFilePath = './../Models/weights.h5'):
    current_file_path = os.path.dirname(__file__)
    json_file = open(os.path.join(current_file_path, modelFilePath), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    RNNModel = model_from_json(loaded_model_json)
    RNNModel.load_weights(os.path.join(current_file_path, weightFilePath))
    return RNNModel