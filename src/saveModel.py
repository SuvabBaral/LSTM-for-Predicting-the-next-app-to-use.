import os
from datetime import datetime


#saving the weights and biases of the trained RNN model.
def saveRNNModel(RNNModel):
    current_file_path = os.path.dirname(__file__)
    currentTime = datetime.now()
    formattedTimeinString = currentTime.strftime('%d%m%Y%H%M%S')
    model_json = RNNModel.to_json()
    os.makedirs(os.path.join(current_file_path, '../Models/' + formattedTimeinString));
    new_file_path = os.path.join(current_file_path, '../Models/' + formattedTimeinString)
    with open(os.path.join(new_file_path, 'trainedModel.json'), "w") as json_file:
        json_file.write(model_json)
    RNNModel.save_weights(os.path.join(new_file_path, 'weights.h5'))