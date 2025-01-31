import pandas as pd
from src.Model import train_model_Linear_Regression
from src.Evaluation import evaluate_model
import joblib
#load the data (X_train, y_train, X_test, y_test)
X_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy.xlsx")
y_train=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_trainy.xlsx")
X_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test.xlsx")
y_test=pd.read_excel("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_test.xlsx")
#Train the model
train_model_Linear_Regression(X_train, y_train, "~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/models/model_v1.pkl")
#Load the training model
model=joblib.load("~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/models/model_v1.pkl")
#Evaluate the model
log_path="results/logs/metrics.txt"
rmse, r2=evaluate_model(model, X_test, y_test, log_path)
#print results
print(f"RMSE: {rmse}, R^2: {r2}")