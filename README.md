# Life-expectancy-analysis-Japan
A Data Science Approach to why Japanese people live for so long having the highest estimated life expectancy at birth of 84.26 years.

# Steps to tackling this Data Science Project

# 1- Problem definition
Define the purpose of the project clearly (this could be adjusted later on as data is being unraveled)
- Insights on factors influencing life expectancy and a prediction model.

# 2- Understanding, cleaning and preprocessing the Data (Check Data_Pre_Processing.py file)
# Steps
1. *Load the raw Data ('/Data/Raw')*
The 1st step to any data science project is to load the raw data into a programming environment like Python in this case using a library like **pandas**
2. *Inspect the Data*
This involves checking and understanding the structure of the dataset
- Check for column names, data types and missing values
- Identify whether all data values are consistent and valid
3. *Handle missing values*
If there are missing values:
- Either drop rows/ columns with significant missing data
- Or impute missing values with the mean, median or mode
4. *Rename columns for clarity*
Rename columns to make sure that they are easy to interpret and work with 
5. *Detect and handle outliers if they exist*
Outliers can distort analysis and modeling and could be due do a sampling error. Use visualizations like boxplots to identify them using **matplotlib** Library. Depending on your interpretation o the finding you could keep the outlier or impute with mean or with nearest highest value.
6. *Encode categorical data* 
Convert non numeric columns ie: categorcial data(Labels) into numerical formats so we can use them with machine learning models like linear regression or decision trees while still preserving information.
For this doing 2 Techniques are available: 
- One-Hot Encoding: Converts each unique category into a new binary column using **pd.get_dummies** and is best used for categories with no inherent order (Country, color)
- Label encoding: Assigns a unique integer to each category using **fit_transform** and is best used for categories that have order or ranking (High school, college..)
7. *Normalize, standardize Numeric Data or Log_trnsform*
Scaling ensures that all numerical features contribute equally to the model therefore making sure that different features (eg. life expectancy in years and salaries in millions) have the same range will ensure consistency with machine learning models that are sensitive to the scale of numerical values (eg. linear regression, NN, clustering). This would involve either Standardization or Normalization or applying a log transformtaion
- Standardization: scale the data to have a mean of 0 and a standard deviation of 1 and is very useful for algorithms that assume normally distributed data like SVM, PCA using **SrandardScaler()**
- Normalization: scale the data to fit within a specific range such as [0, 1] and is very Useful for when absolute values matter eg. distance based algorithm like KNN using **MinMaxScaler()**
-Log transform: apply logarithmic function to the data to handle skewness, and compress large ranges.
**log()**
8. *Save the cleaned data* 
Store the cleaned and processed dataset for future use (Check Cleaned_Japan_Life_Expectancy.xlsx file)

# 3- EDA (Exploratory Data Analysis) using JupyterNotebook (check exploratory_analysis.ipynb file)
- Generate summary statistics like mean, min, max and standard deviation
- Plot distributions of features using Histograms to identify skewness and patterns
- Visualize correlations between features (x) and targets (y) using a heatmap (~+1 Strong, ~-1 Neg_strong, 0 No_correlation) and feature interaction using scatter plots 
- Pair plots for key features to visualize the relationship between multiple key features simultaneously
- Identify erros, outliers and anolmalies using Boxplots
- Identify strong and weak correlations between features and targets using heat map and plot pairing
- Use variance Inflation Factor (VIF) test to ensure no severe multicolinearity exists between features(~1 good, >10 High)

# 4- Feature engineering (Check Data_Pre_Processing.py file)
Transform the data to improve model performance by:
- Selecting the relevant features (Independant Variable) based on correlation analysis(Correlation Heatmap) and the target (Dependant Variable) while avoiding multicollinearity (features that are highly correlated may impact the model negatively)
- Creating derived features if necessary like PCA (principal component analysis) to combine features

# 5- Data splitting (Check Data_Pre_Processing.py file)
split the data into training data and test data to ensure that once the model is ready it can be evaluated on unseen data(test data) and save the csv file in processed data folder
- Training set = 80% of the data
- Testing set = 20% of the data 

# 6- Model Selection and training Function (Check Model.py file)
To predict the Target based on the relevant features we need to choose a suitable algorithm and train the model based on the data type:
- **Regression**: Continuous Target (eg, life expectancy); for small datasets and linear relationships use *Linear regression*, for Large Datasets and non linear relationships use more complex models like *Decision Trees*, *Random Forest* or *Gradient Boosting*
- **Classification**: Discrete labels (eg, Predict whether a customer buys a product or not); if relationship is linear *Logistic Regression*, low dimensional feature space then *K-Nearest-Neighbor*, Non linear relationship then *Decision Tree* or *Random Forest*, High dimensional feature space *Support Vector Machines*

# 7- Model Evaluation (Check Evaluation.py file)
Using the testing data we will evaluate how well the model prdicts the target value.
- Plot the predicted target points vs the actual target points for the testing data set and check how accurate the trained model was at predicting the target value. (Points should be as close as possible to the diagonal line)
- For regression Models: 
  * R^2 (R squared): *Coefficient of determination* indicates how well the key features chosen (independant variables) were able to predict the target. (r2=1: perfect fit, r2=0: Model performs very poorly)
  * RMSE (Root Mean Squared Error): Average error between predicted values and actual values. (The lower the better) 
-For Classification Models:
  *Accuracy Score: Measures the proportion of correct predictions


# 8- Feature importance and Visualization (Check feature_importance.ipynb file)
Use the trained model to quantify which features had the most influence on its predictions. We will rank features by importance to provide further insight. If needed further opeartions should be perfomed in Data_pre_processing.py
- Access coefficent as feature importance for Linear Regression: retrieve cofficient for each feature and rank them by order of magnitude to discover feature importance (+Coef increase target value, -Coef decrease target value, Mangitude indicates importance). Limitations: non scaled features may be misleading, overlapping features
- SHAP: works for any machine learning model and calculates how much each feature contributes to the prediction by comparing the model's output with and without the feature (+SHAP increase prediction, -SHAP decrease prediction and the vertical axis lists the features in order of importance while on the Horizontal axis the further from 0 the stronger the impact)
- Interpretation: use visualizations (dependance plot, scatterplot, heatmap..) and aggregate the data into bins to dive deeper and to understand and validate feature interactions

# 9- Model Optimization and evaluation (Check model_experiments.ipynb file)
To improve the performance and accuracy of the model by *fine-tuning* its hyperparameters (Hyperparameters are the settings of a model that are not learned during training but need to be set before the learning process begins) and reduce overfitting or underfitting (too many or too few features) using *manual search, grid search, random search*..
Evaluate the tuned model on test data and compare performance with baseline model

# 10- Deployment and Reporting (Check observations.md)
- Summarize results, findings, Insights and data driven recommendations for stakeholders (results/observations/) and save *prediction model*(models/prediction_model.pkl) and *prediction values* to excel file (results/predictions/predicted_values.excel) and *scatter plot of actual vs predicted model values*(results/predictions/scatter_plot_actual_vs_predicted_values.png)
- Deploy the model using tools like *Flask, Streamlit* for live predictions