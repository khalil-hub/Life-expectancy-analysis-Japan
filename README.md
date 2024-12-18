# Life-expectancy-analysis-Japan
A Data Science Approach to why Japanese people live for so long having the highest estimated life expectancy at birth of 84.26 years.

# Steps to tackling this Data Science Project

# Steps to cleaning and preprocessing the Data (Check Data_Pre_Processing.py file)
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
7. *Normalize or standardize Numeric Data*
Scaling ensures that all numerical features contribute equally to the model therefore making sure that different features (eg. life expectancy in years and salaries in millions) have the same range will ensure consistency with machine learning models that are sensitive to the scale of numerical values (eg. linear regression, NN, clustering). This would involve either Standardization or Normalization
- Standardization: scale the data to have a mean of 0 and a standard deviation of 1 and is very useful for algorithms that assume normally distributed data like SVM, PCA using **SrandardScaler()**
- Normalization: scale the data to fit within a specific range such as [0, 1] and is very Useful for when absolute values matter eg. distance based algorithm like KNN using **MinMaxScaler()**
- why do we put double [[]] for the column
8. *Split data into features and targets*
seperate the independant variables 'x'(features) from the dependant variables (target)'y' for analysis or modeling
?? how can we do multiple columns on x axis
9. *Save the cleaned data* 
Store the cleaned and processed dataset for future use