# Predicting-the-next-app-to-use.

This is a Recurrent neural network model that helps to predict the next app a user might use based on his previous app usage data.
The Dataset contains around 2000 time-based app usage data of the user in a week.
Since only 36 apps were used by the user in that particular week, the prediction is limited to the 36 apps.

Since I have already trained the model , " latest.h5" file contains the pre-trained weights for the model , so that you no need to re-train the model evereytime.
"latest.json" file stores the model structure.

The "model2.py" program can be directly run by , first importing the dataset, then loading the "latest.h5"(weights) and "latest.json"(model)
file.(which is present towards the end of the program), or you can run for yourself by trainning the model again.

The same "dataset.csv" is divided into training and test data.
Since our mobile phone's usually contain around 4 app suggestion space, we take the first 4(with the highest probability) predicted apps ,to test the accuracy of our model.

"actual_app_used" variable is a dataframe that contains the actual apps that were used by the user. And 
"prediction" variable is a dataframe that  contains the predicted apps of the model during the same instance.
