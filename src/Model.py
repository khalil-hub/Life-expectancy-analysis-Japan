import joblib
from sklearn.feature_selection import RFE
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
    #Perform RFE (Recursive Feature Elimination) to select top 2 features
    rfe=RFE(estimator=model, n_features_to_select=2)
    rfe.fit(X_train, y_train)
    #print selected features
    selected_features=X_train.columns[rfe.support_]
    print(f"Selected features: {selected_features}")
    #Save the trained model
    joblib.dump(rfe, model_path)
    print(f"Model saved to {model_path}")
    