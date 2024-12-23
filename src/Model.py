import joblib
#Model Selection Training
#Linear Regression(Linear Relationship between Features and Target)
from sklearn.linear_model import LinearRegression
def train_model_Linear_Regression(X_train, y_train, model_path):
    """ X_train: Feature matrix for training (2D array-like).
    - y_train: Target values for training (1D array-like).
    - model_path: Path where the trained model will be saved                      
      """
    #Model Training: Linear Regression
    model=LinearRegression()
    model.fit(X_train, y_train)
    #Save the trained model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
