import pandas as pd
import numpy as np
from src.Model import train_model
from src.Evaluation import evaluate_model, save_predictions
from src.Grid_Search import best_model

import joblib
#load the data (X_train, y_train, X_test, y_test)
X_train=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_train.xlsx')
X_test=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_test.xlsx')
y_train=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_train.xlsx')
y_test=pd.read_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_test.xlsx')

#Model Dictionary
model_paths={"Lasso Regression": "models/model_Lasso_Regression.pkl",
             "Random Forest Regressor": "models/model_Random_Forest.pkl",
             "XGBoost": "models/model_XGBoost.pkl",
             "Support Vector Regression": "models/model_Support_Vector.pkl"
}
X_train_=X_train.drop('Socioeconomic_index', axis=1)
X_test_=X_test.drop('Socioeconomic_index', axis=1)

for model_name, model_path in model_paths.items():
    #Train the model and save it
    train_model(X_train_, y_train, model_name, model_path)
    #Load the training model
    model=joblib.load(model_path)
    print(f"Loaded {model_name} from {model_path}")
    #Evaluate the model and log the evaluation metrics
    log_path=f"results/logs/{model_name.replace(' ', '_').lower()}_metrics.txt"
    rmse, r2, y_pred=evaluate_model(model, X_test_, y_test, log_path)
    #print results
    print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")
    #Save the predicted vs actual data to excel and the scatterplot
    prediction_path=f"results/predictions/{model_name.replace(' ', '_').lower()}_predicted_vs_Actual_life_expectancy.xlsx"
    save_predictions(y_test, y_pred, prediction_path)
    print(f"Predictions for {model_name} saved to {prediction_path}")

#perform grid search to determine best model hyperparameter
best_model(X_train, y_train, "models/")
model_paths={"Lasso Regression": "models/Grid_search_Lasso_Regression.pkl",
             "Random Forest Regressor": "models/Grid_search_Random_Forest_Regressor.pkl",
             "Support Vector Regression": "models/Grid_search_Support_Vector_Regression.pkl"
}
for model_name, model_path in model_paths.items():
    model=joblib.load(model_path)
    log_path=f"results/logs/Grid_search_{model_name.replace(' ', '_').lower()}_metrics.txt"
    rmse, r2, y_pred=evaluate_model(model, X_test, y_test, log_path)

#linear regression post PCA 
model_name="linear_regression_PCA"
model_path="models/model_linear_regression_pca.pkl"
X_train_PCA=X_train.drop(['University_%', 'Salary', 'Elementary_school'], axis=1)
X_test_PCA=X_test.drop(['University_%', 'Salary', 'Elementary_school'], axis=1)
train_model(X_train_PCA, y_train, model_name, model_path)
model=joblib.load(model_path)
print(f"Loaded {model_name} from {model_path}")
log_path=f"results/logs/{model_name}_metrics.txt"
rmse, r2, y_pred=evaluate_model(model, X_test_PCA, y_test, log_path)
print(f"{model_name} - RMSE: {rmse:.2f}, R^2: {r2:.2f}")
prediction_path=f"results/predictions/{model_name}_predicted_vs_Actual_life_expectancy.xlsx"
save_predictions(y_test, y_pred, prediction_path)
print(f"Predictions for {model_name} saved to {prediction_path}")