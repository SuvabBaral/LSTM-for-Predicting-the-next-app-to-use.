# Predicting-the-next-app-to-use.

This is a LSTM(Long-Short Term Memory)-Recurrent neural network model that helps to predict the next app a user might use based on his previous app usage data.
The Dataset contains around 2000 datapoints reflecting all the applications that were used by a user within 1 week.

Since only 36 apps were used by the user in that particular week, the prediction is limited to the 36 apps.

To reuse the already trained model:

```python src/generateModel.py True```

To generate and train a new model:

```python src/generateModel.py```

The same "dataset.csv" is divided into training and test data.
Since our mobile phone's usually contain around 4 app suggestion space, we take the first 4(with the highest probability) predicted apps ,to test the accuracy of our model.

"actual_app_used" variable is a dataframe that contains the actual apps that were used by the user. And 
"prediction" variable is a dataframe that  contains the predicted apps of the model during the same instance.


Final Output-
<img src="https://github.com/SuvabBaral/LSTM-for-Predicting-the-next-app-to-use./blob/master/Output.png" width="800" height="390" title="Output">
