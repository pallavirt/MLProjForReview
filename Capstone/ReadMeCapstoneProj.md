*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Capstone Project : Azure Machine Learning model for Fraud detection

This project creates a machine learning model for detecting fraud in card transactions. It uses 2 different approaches to train the model:
1. Using AutoML
2. Using Hyperdrive

The input dataset needed for the training is downloaded from Kaggle.

## Dataset

### Overview
The dataset for Fraud detection in card payment is downloaded from Kaggle. It's a single CSV file with nearly 1,50,000 records. A dataset named KaggleDSFraudDS is created in workspace based on that. 

### Task
This dataset is used to predict if the given transaction is legit or fraud. The model is trained using LogisticRegression algorithm. legit is the positive label.

### Access
It's created as a dataset in the datastore in the current workspace
![image](https://user-images.githubusercontent.com/109726862/184794471-4fd89625-b1fc-4640-9c66-705b960cf285.png)

## Automated ML
The automl model is trained for classification scenario. The experiment will run maximum for 20 minutes. It uses the given dataset and is trained to predict the class for column EVENT_LABEL. It uses AUC_Weighted metric for the fraud detection scenario.
![image](https://user-images.githubusercontent.com/109726862/184794873-f7d5707e-eced-483e-be02-a0055d1c0ee6.png)

### Results
The best model found with AutoML used algorithm 'XGBoostClassifier'. The AUC_Weighted value was 0.956. The best model was then registered in the workspace.
![image](https://user-images.githubusercontent.com/109726862/184794892-c6e9e1bd-dba6-41b2-9f08-4579cb2be3c8.png)
![image](https://user-images.githubusercontent.com/109726862/184794913-4d22ae1c-7f0a-4eb8-99b0-d3c57ece7c81.png)
![image](https://user-images.githubusercontent.com/109726862/184795272-e9bd7b51-15cb-43e7-9b8c-3ec9978c2b04.png)

## Hyperparameter Tuning
The code tries to find the best values for hyperparameters, Inverse of regularization (C) and Max_Iter for Logistic Regression algorithm. In order to reduce the time taken while running multiple iterations, it uses BanditPolicy. BanditPolicy helps in early termination of low-performing long running jobs. The model training script is provided in Train.py file. The script reads the data file from the dataset created in the current workspace. It then removes certain columns which are not necessary for the model creation. It also converts categorical columns into dummy numeric indicators. This is required for LogisticRegression algorithm. 
![image](https://user-images.githubusercontent.com/109726862/184795228-6425a179-031a-4173-9600-f376652f2f5e.png)

### Results
The best model was found for the hyperparameter values - C=0.001, max_iter=100
The corresponding value for metric, Accuracy is 0.958
The best model was also registered in the current workspace
![image](https://user-images.githubusercontent.com/109726862/184795252-43417e5b-56cb-447d-80fb-f4402196beb5.png)

## Model Deployment
The best run found with AutoML was registered as a model and deployed as a service to Azure Container Instance. The deployment involved creating a suitable environment yml file, a scoring python script with init and run method. These details were provided using inference config and deployment config. The service endpoint was finally tested with a sample test record.
![image](https://user-images.githubusercontent.com/109726862/184795463-5c11be3b-ab69-4ff8-bb52-f3a388e452b4.png)
![image](https://user-images.githubusercontent.com/109726862/184795484-ef7257f1-2ad2-403a-b718-d5f602267fd4.png)

As shown in capstoneautoml.ipynb file, the best run was registered as a model in the workspace and deployed as a service. The notebook also contains a sample JSON input and call to the deployed service using requests.post method. It has captured the returned response as well. 
![image](https://user-images.githubusercontent.com/109726862/184937049-040a3c2d-f8fe-4922-b596-ff862c1942fe.png)

## Further Enhancements:

1. The data preprocessing can be enhanced to include the appropriate features.

2. Regarding HyperDrive experiment,

  * More runs can be added.
  * Some ensemble method can be used instead of logistic regression to see if that improves the model performance further.
  

## Screen Recording
https://youtu.be/s9-WOQFPgVs

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
