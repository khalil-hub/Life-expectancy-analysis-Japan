import pandas as pd
import numpy as np
from src.Model import train_model
from src.Evaluation import evaluate_model, save_predictions

import joblib
#load the data (X_train, y_train, X_test, y_test)
X_train=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_train.xlsx')
X_test=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_test.xlsx')
y_train=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_train.xlsx')
y_test=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_test.xlsx')

#Model Dictionary
model_paths={"Linear Regression": "models/model_Linear_Regression.pkl",
             "Lasso Regression": "models/model_Lasso_Regression.pkl",
             "Random Forest Regressor": "models/model_Random_Forest.pkl",
             "XGBoost": "models/model_XGBoost.pkl",
             "Support Vector Regression": "models/model_Support_Vector.pkl"
}

for model_name, model_path in model_paths.items():
    #Train the model and save it
    train_model(X_train, y_train, model_name, model_path)
    #Load the training model
    model=joblib.load(model_path)   
    print(f"Loaded {model_name} from {model_path}")
    #Evaluate the model and log the evaluation metrics
    log_path=f"results/logs/{model_name.replace(' ', '_').lower()}_metrics.txt"
    rmse, r2, y_pred=evaluate_model(model, X_test, y_test, log_path)
    #print results
    print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")
    #Save the predicted vs actual data to excel and the scatterplot
    prediction_path=f"results/predictions/{model_name.replace(' ', '_').lower()}_predicted_vs_Actual_life_expectancy.xlsx"
    save_predictions(y_test, y_pred, prediction_path)
    print(f"Predictions for {model_name} saved to {prediction_path}")
