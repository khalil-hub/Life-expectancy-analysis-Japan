from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.svm import SVR
import joblib
def best_model(X_train, y_train, model_path):
# Define the models and their parameter grids
    param_grids = {
        "Lasso Regression": {
            "model": Lasso(),
            "params": {"alpha": [0.01, 0.1, 1, 10, 100]}
        },
        "Random Forest Regressor": {
            "model": RandomForestRegressor(),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Support Vector Regression": {
            "model": SVR(),
            "params": {
                "C": [0.1, 1, 10],
                "epsilon": [0.01, 0.1],
                "kernel": ["linear", "rbf"]
            }
        }
    }
    #perform grid search
    best_models={}
    for model_name, config in param_grids.items():
        grid_search=GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_models[model_name]=grid_search.best_estimator_
        print(f"Best params for {model_name}: {grid_search.best_params_}")
        #save best model with best params
        joblib.dump(grid_search.best_estimator_, f"{model_path}Grid_search_{model_name.replace(' ', '_')}.pkl")
    return best_models
