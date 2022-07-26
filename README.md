# Optimizing an ML Pipeline in Azure
## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.
## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)
## Summary
This dataset contains details of people who participated in different marketing campaigns conducted by a bank. With this data, we seek to predict who would be potential customers for any future schemes of the bank.

The best performing model was found using **AutoML method and the best performing algorithm was VotingEnsemble** for the given dataset.
## Scikit-learn Pipeline
To achieve this, we have used Scikit-learn Logistic Regression algorithm. Following are the different activities in the pipeline:
1. A dataset is created using the CSV file available at the given URL. 
2. The dataset is cleaned using the given clean_data function
3. The dataset is then split into training and testing data.
4. The model is created by using Scikit learn Logistic Regression algorithm, training data and different values for the given hyperparameters. 

**Hyperparameters used:**
Inverse of regularization (C) and Max_Iter

Hyperparameter values are selected randomly using **RandomSampling** method. RandomSampling method is faster, and it also supports early termination policy.

Early termination policy: **BanditPolicy** is used to avoid wasting time in running a large number of runs which may not result into best runs. It compares the target performance metric of a run with the similar metric of the best run. It stops the run if it underperforms the best run.

5. The accuracy value is logged for each run.

![image](https://user-images.githubusercontent.com/109726862/180614709-eb735ae5-a497-41b3-9883-e9fa395021ed.png)

The **highest accuracy value observed is 0.913 with C value as 0.001 and max_iter value as 100**, as shown in above screenshot.

## AutoML
**In 1-2 sentences, describe the model and hyperparameters generated by AutoML.**

AutoML pipeline included the following activities:
1.	It also started with creation of the dataset using TabularDatasetFactory.
2.	The dataset was cleaned up using the given clean_data function.
3.	Configuration used for AutoML experiment:
	experiment_timeout_minutes=30,
    	task='classification',
    	primary_metric='accuracy',
    	training_data=ds,
    	label_column_name='y',
    	n_cross_validations=5
4.	**Highest accuracy achieved with AutoML: 0.918, Algorithm used: VotingEnsemble**
![image](https://user-images.githubusercontent.com/109726862/180614832-0ba71bc3-866c-4352-8f31-dbca9ff19749.png)
 
## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

Accuracy value achieved with AutoML is slightly better than that achieved with Hyperdrive. Additionally, AutoML reduces the efforts required for any manual adjustments of hyperparameters.

## **Metrics for AutoML:**
![image](https://user-images.githubusercontent.com/109726862/180614866-da0177d4-98c0-4a2d-a0c7-b0fa290db645.png)

## **Metrics for Hyperdrive:**
![image](https://user-images.githubusercontent.com/109726862/180614729-92fb70ba-7b6f-477b-8366-e3be3473deee.png)
 

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

To try out different metrics other than Accuracy and see if that brings any further improvement.




