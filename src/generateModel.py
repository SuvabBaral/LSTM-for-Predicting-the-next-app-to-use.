# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:37:35 2019

@author: Suvab Baral
"""

import sys
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from getModel import getTrainedModel
from saveModel import saveRNNModel

#removing the redundant data.
def clean_data(data):
    data=data[~(data == 'Screen off (locked)').any(axis=1)]
    data=data[~(data == 'Screen on (unlocked)').any(axis=1)]
    data=data[~(data == 'Screen off (unlocked)').any(axis=1)]
    data=data[~(data == 'Screen on (locked)').any(axis=1)]
    data=data[~(data == 'Screen on').any(axis=1)]
    data=data[~(data == 'Screen off').any(axis=1)]
    data=data[~(data == 'Device shutdown').any(axis=1)]
    data=data[~(data == 'Device boot').any(axis=1)]
    data=data.dropna()
    data.index=range(len(data))
    return data

def encode_data(data):
    label_encoder_app=LabelEncoder()
    encoded_data=label_encoder_app.fit_transform(data.iloc[:,0:1])
    encoded_data=pd.DataFrame(data=encoded_data)
    return [encoded_data, label_encoder_app]

#splitting the training and testing data.
def split_into_train_test_set(encoded_data):
    train_set=encoded_data.iloc[:1901,0:1].values
    test_set=encoded_data.iloc[1901:,0:1].values
    return [train_set, test_set]

def getRNNModel(X_train, use_preTrained_model):
    if (use_preTrained_model == 'True'):
        print('***************Using pretrained model***************')
        RNNModel = getTrainedModel()
    else:
        print('***************Creating model and start training***************')    
        RNNModel = Sequential()

        RNNModel.add(LSTM(units = 200,return_sequences = True, input_shape=(X_train.shape[1], 1)))
        RNNModel.add(Dropout(rate=0.3))

        RNNModel.add(LSTM(units =200, return_sequences = True))
        RNNModel.add(Dropout(rate=0.3))
        
        RNNModel.add(LSTM(units =200, return_sequences = True))
        RNNModel.add(Dropout(rate=0.3))

        RNNModel.add(LSTM(units = 200, return_sequences = True))
        RNNModel.add(Dropout(rate=0.3))

        RNNModel.add(LSTM(units =200, return_sequences = False)) #False caused the exception ndim
        RNNModel.add(Dropout(rate=0.3))

        RNNModel.add(Dense(units= 36,activation='sigmoid'))

    RNNModel.compile(optimizer = 'rmsprop', loss = 'mean_squared_error', metrics = ['accuracy'])
    return RNNModel

use_preTrained_model =  False
if (len(sys.argv) > 1):
    use_preTrained_model = sys.argv[1]

data=pd.read_csv('../Datasets/dataset.csv')
data = clean_data(data)
[encoded_data, label_encoder_app] = encode_data(data)
[train_set, test_set] = split_into_train_test_set(encoded_data)

scaler=MinMaxScaler(feature_range=(0,1))
training_set_scaled=scaler.fit_transform(train_set)

# Transoform data to easily feed into the model
X_train=[]
y_train=[]

for i in range(10,1901):
    X_train.append(training_set_scaled[i-10:i,0])
    y_train.append(train_set[i,0])

X_train=np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

label_encoder_y=LabelEncoder()
y_train=label_encoder_y.fit_transform(y_train)
y_train=np.array(y_train)
y_train= keras.utils.to_categorical(y_train, num_classes=36)


#training
RNNModel = getRNNModel(X_train, use_preTrained_model)
if (use_preTrained_model != 'True'):
    RNNModel.fit(X_train, y_train, epochs = 150, batch_size = 16)
    # print(RNNModel.history.history['acc'])
# print(RNNModel.summary())


#testing
total_dataset=encoded_data.iloc[:,0:1]
inputs=total_dataset[len(total_dataset)-len(test_set)-10:].values
inputs=inputs.reshape(-1,1)
inputs=scaler.transform(inputs)
decoded_input=scaler.inverse_transform(inputs)
X_test=[]
for i in range(10,397):
    X_test.append(inputs[i-10:i, 0])
X_test=np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
predicted_app=RNNModel.predict(X_test)
predicted_app_1=RNNModel.predict_classes(X_test)

#predicted_app_1=sc.inverse_transform(predicted_app)
idx = (-predicted_app).argsort() 

#confusion matrix (just for chekcking the accuracy)
cm=np.zeros(shape=(2,2))
for i in range(387):
    if(test_set[i]== idx[i,0] or test_set[i]==idx[i,1] or test_set[i]==idx[i,2] or test_set[i]==idx[i,3] ):
        cm[1,1]+=1
    else:
        cm[1,0]+=1

#index of the highest values.
idx=pd.DataFrame(idx)
prediction=label_encoder_app.inverse_transform(idx.iloc[:,0:1])
prediction=pd.DataFrame(data=prediction)
actual_app_used=label_encoder_app.inverse_transform(test_set)
actual_app_used=pd.DataFrame(data=actual_app_used)

for i in range(1,4):
    idx3=label_encoder_app.inverse_transform(idx.iloc[:,i:i+1])
    idx3=pd.DataFrame(data=idx3)    
    prediction=pd.concat([prediction,idx3], ignore_index=True, axis=i)
#if we want the prediction of the entire 36 apps. USE ' range (1,36) ' in the for loop

final_outcome = pd.concat([prediction, actual_app_used], axis = 4)
final_outcome.columns = ['Prediction1', 'Prediction2', 'Prediction3', 'Prediction4', 'Actual App Used']
print('***********************************FINAL PREDICTION*********************************')
print(final_outcome)

if (use_preTrained_model != 'True'):
    final_model_accuracy = RNNModel.history.history['acc']
    print('Accuracy of the model: ', round(final_model_accuracy[0] * 100, 2))
    saveRNNModel(RNNModel)







