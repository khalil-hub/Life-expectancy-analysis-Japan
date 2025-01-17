import pandas as pd
import numpy as np
from src.Model import train_model_Linear_Regression
from src.Evaluation import evaluate_model, save_predictions

import joblib
#load the data (X_train, y_train, X_test, y_test)
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_train.xlsx")
y_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_train.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test.xlsx")
y_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_test.xlsx")

#Train the model and save it 
train_model_Linear_Regression(X_train, y_train, "models/model_v_test.pkl")
#Load the training model
model=joblib.load("models/model_v_test.pkl")
#Evaluate the model and log the evaluation metrics
log_path="results/logs/metrics_test.txt"
rmse, r2, y_pred=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")
#Save the predicted vs actual data to excel and the scatterplot
save_predictions(y_test, y_pred, "results/predictions/predicted_vs_Actual_test.xlsx")