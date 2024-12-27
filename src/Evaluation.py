import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error, r2_score
import os
def evaluate_model(model, X_test, y_test, log_path):
#Model Evaluation
    """Evaluates the performance of a trained model using the test data.
    
    Parameters:
    - model: The trained machine learning model.
    - X_test: Feature matrix for testing (2D array-like).
    - y_test: Target values for testing (1D array-like).
    
    Returns:
    - rmse: Root Mean Squared Error of the model on the test data.
    - r2: R² score of the model on the test data. 
    """
    #Make prediction using the test data
    y_pred=model.predict(X_test)
    plt.figure(figsize=(8,6))
    #scatter plot of actual vs predicted 
    plt.scatter(y_test, y_pred, color='b', label='Actual vs Predicted')
    #add title and labels
    plt.title('Linear Regression: Actual vs Predicted Life Expectancy')
    plt.xlabel('Actual Life Expectancy')
    plt.ylabel('Predicted Life Expectancy')
    #plot the diagonal line(perfect prediction line)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
    #Show the plot
    plt.legend()
    plt.show()
    #Calculate RMSE (Root Mean Square Error)
    rmse=root_mean_squared_error(y_test, y_pred)
    #calculate R^2
    r2=r2_score(y_test, y_pred)
    #Log the results to a file
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as f:
        f.write(f"Root mean square error (RMSE): {rmse}\n")
        f.write(f"R^2 Score: {r2}\n")
    
    print("Evaluation metrics logged to {log_path}")

    return rmse, r2