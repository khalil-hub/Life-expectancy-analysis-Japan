import pandas as pd
import numpy as np
from src.Model import train_model_Linear_Regression
from src.Evaluation import evaluate_model
import joblib
#load the data (X_train, y_train, X_test, y_test)
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy.xlsx")
y_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_trainy.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test.xlsx")
y_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_test.xlsx")

#Train the model 
train_model_Linear_Regression(X_train, y_train, "models/model_v1.pkl")
#Load the training model
model=joblib.load("models/model_v1.pkl")
#Evaluate the model
log_path="results/logs/metrics.txt"
rmse, r2=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")

#Train the model with Log transformation 
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_log.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_log.xlsx")
train_model_Linear_Regression(X_train, y_train, "models/model_v2.pkl")
#Load the training model
model=joblib.load("models/model_v2.pkl")
#Evaluate the model
log_path="results/logs/metrics_Log_transform.txt"
rmse, r2=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")

#Train the model with Log transformation and elemetary school
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_v2.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_v2.xlsx")
train_model_Linear_Regression(X_train, y_train, "models/model_v3.pkl")
#Load the training model
model=joblib.load("models/model_v3.pkl")
#Evaluate the model
log_path="results/logs/metrics2.txt"
rmse, r2=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")

#Train the model with Log transformation and elemetary school
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_RFE.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_RFE.xlsx")
train_model_Linear_Regression(X_train, y_train, "models/model_v4.pkl")
#Load the training model
model=joblib.load("models/model_v4.pkl")
#Evaluate the model
log_path="results/logs/metrics_RFE.txt"
rmse, r2=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")