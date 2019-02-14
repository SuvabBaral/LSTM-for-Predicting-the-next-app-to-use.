# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:37:35 2019

@author: Suvab Baral
"""

import pandas as pd
import numpy as np
#removing the redundant data.
data=pd.read_csv('dataset.csv')
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

from sklearn.preprocessing import LabelEncoder
label_encoder_app=LabelEncoder()
data2=label_encoder_app.fit_transform(data.iloc[:,0:1])
data2=pd.DataFrame(data=data2)
 
#splitting the training and testing data.

train_set=data2.iloc[:1901,0:1].values
test_set=data2.iloc[1901:,0:1].values


from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_scaled=sc.fit_transform(train_set)

X_train=[]
y_train=[]
for i in range(10,1901):
    X_train.append(training_set_scaled[i-10:i,0])
    y_train.append(train_set[i,0])
X_train=np.array(X_train)

label_encoder_y=LabelEncoder()

y_train=label_encoder_y.fit_transform(y_train)
y_train=np.array(y_train)

import keras
y_train= keras.utils.to_categorical(y_train, num_classes=36)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


#train
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()
regressor.add(LSTM(units = 200,return_sequences = True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(rate=0.3))

regressor.add(LSTM(units =200, return_sequences = True))
regressor.add(Dropout(rate=0.3))
regressor.add(LSTM(units =200, return_sequences = True))
regressor.add(Dropout(rate=0.3))

regressor.add(LSTM(units = 200, return_sequences = True))
regressor.add(Dropout(rate=0.3))

regressor.add(LSTM(units =200, return_sequences =False))#False caused the exception ndim
regressor.add(Dropout(rate=0.3))



regressor.add(Dense(units= 36,activation='sigmoid'))
regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs = 150, batch_size =16)




#test
total_dataset=data2.iloc[:,0:1]
inputs=total_dataset[len(total_dataset)-len(test_set)-10:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)
decoded_input=sc.inverse_transform(inputs)
X_test=[]
for i in range(10,397):
    X_test.append(inputs[i-10:i, 0])
X_test=np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


    
predicted_app=regressor.predict(X_test)
predicted_app_1=regressor.predict_classes(X_test)


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
    prediction=pd.concat([prediction,idx3],ignore_index=True,axis=i)
#if we want the prediction of the entire 36 apps. USE ' range (1,36) ' in the for loop






#saving the weights and biases of the trained RNN model.
model_json = regressor.to_json()
with open("latest.json", "w") as json_file:
    json_file.write(model_json)
regressor.save_weights("latest.h5")


#load the trained RNN model
from keras.models import model_from_json
json_file = open('latest.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
regressor= model_from_json(loaded_model_json)
regressor.load_weights("latest.h5")







