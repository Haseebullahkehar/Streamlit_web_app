# Prompt 

Hey Chat GPT, act as a  application developer expert in python using streamlit,
and build an machine learning application using scikit learn with the following workflow 

1.  Greet the user with a welcome message and a brief 
description of the application.
2.  Ask the user if he wants to upload the data or 
use the example dataset.
3. If the user select to upload the data, show the uploader section on the sidebar, upload the dataset in csv, xlsx, tsv or any other possible data format.
4. If the user do not want to upload the dataset then provide a default dataset selection box on the sidebar.This selection box should download the dataset from the sn.load_dataset function. The dataset should include Datasets include titanic ,tips  or iris. 
5.  Print the basic data information such as data head, data shape, data description, and data info and column name. 
6. Ask from the user to select the columns as features and the columns as target 
7. Identify the problem if the target column is continuos numeric column then print the message that this is regression problem or classification problem.
8. Pre-process the data, if the data contains any missing values then fill the missing 
values with iterative imputer function of scikit-learn, if the features are not in the same scale then scale the features using the standard scaler function of scikit-learn. If the features are categorical variables then encode the categorical variables using the label encoder function of scikit-learn. Please bear in mind to keep the encoder separate for each column as we need to inverse transform the data at the end.
9. Ask the user to provide the train test split via sluder or user input function.
10. Ask the user to select the model from the sidebar, the model should include linear regression, decision tree, the random forest and support vector machines and same classes of models for the classification.
11. Train the models on the training data and evaluate on the test data 
12. if the problem is a regression problem, use the mean squared error, RMSE, MAE, AUROC Ccurveand r2 score for the evaluation ,if the problem if of classification, use the accuracy score, precision, recall, f1 score and draw confusion matrix for evaluation.
13. Print the evaluation matrix for each model.
14. Highlight the best model based on the evaluation matrix. 
15. Ask the user if he wants to download the model , if yes then download the model in the pickle format.
16. Ask the user if he wants to make the prediction, if yes then aske the user to provide the input data using slider or uploader file and make the prediction using the best model.
17. Show the prediction to the user.

 