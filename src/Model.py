import joblib
#Model Selection Training
#Linear Regression(Linear Relationship between Features and Target)
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

def train_model(X_train, y_train, model_name, model_path):
  """ X_train: Feature matrix for training (2D array-like).
  - y_train: Target values for training (1D array-like).
  - model_path: Path where the trained model will be saved                      
  """
  #Models to be used
  #Lasso regression to perform feature selection (small alpha behaves like LR)
  #Random forest Regressor, XGBoost, Support vector regression to capture non linear relationships
  
  models={
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1, random_state=42),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
    "Support Vector Regression": SVR(kernel='rbf', C=1.0, epsilon=0.1)
  }  
  #Check if the selected model exists
  if model_name not in models: 
    raise ValueError(f"Model {model_name} not found. Available Models: {list(models.keys())}")
  #Select the model
  model=models[model_name]
  #Model Training
  print(f"Training Model:{model_name}")
  model.fit(X_train, y_train)
  #Save the trained model
  joblib.dump(model, model_path)
  print(f"Model {model_name} saved to {model_path}")

