import pandas as pd
import numpy as np
# Understanding, cleaning and preprocessing the Data (Check Data_Pre_Processing.py file)

#Step1: Load the data
#Load the Raw excel file using the directory of the file and read the file using pandas library excel reading function
raw_data_path='~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/raw/Japan_Life_Expectancy.xlsx'
data=pd.read_excel(raw_data_path)

#Display the columns
print(data.columns)
#Display the first n rows 
print(data.head(10))

#Step2: Inspect the data
#Check for column names, types and missing data 
print(data.info())
#Get some insight on your data column such as mean, std, min, max
print(data.describe())

#Step3:Handle missing values 
#Check for missing values
print(data.isnull().sum())
#Fill missing values with column means (if applicable, not the case here)
#data_filled=data.fillna(data.mean())
#Alternatively if significant missing data then drow rows
#data_cleaned=data.dropna()

#step4: Rename columns for clarity
#rename columns
data.rename(columns={'Physician': 'Physician_100kP', 'Junior_col': 'Junior_col_%', 'University': 'University_%', 'Public_Hosp': 'Public_Hosp_%', 'Pshic_hosp': 'Psych_Hosp_100kP', 'Beds_psic': 'Psych_Beds_100kP', 'Nurses': 'Nurses_100kP', 'Avg_hours': 'Avg_Work_Hours_Month', 'Elementary_School': 'Elementary_School_%', 'Sport_fac': 'Sport_fac_1MP', 'Park': 'Park_Land_%', 'Forest': 'Forest_Land_%', 'Income_per capita': 'Income_capita', 'Density_pop': 'Population_Density_km2', 'Hospitals': 'General_Hospital_100kP', 'Beds': 'General_Hospital_Beds_100k', 'Ambulances': 'Ambulances_100kP', 'Health_exp': 'Health_Expenditure_%', 'Educ_exp': 'Educ_Expenditure_%', 'Welfare_exp': 'Welfare_Expenditure_%'}, inplace=True)
print(data.info())

#step5: Detect and Handle outliers if they exist
import matplotlib.pyplot as plt

#visualize outliers in the life expectancy column for example
plt.boxplot(data['Life_expectancy'])
plt.title("Outliers in Life expectancy")
plt.show()

plt.boxplot(data['Salary'])
plt.title("Outliers in Life expectancy")
plt.show()
#For life expectancy Aomori is outside of the boxplot and for Salary Tokyo is outside of the boxplot but since it doesnt appear to be a data entry error and doesn't significanty skew our data I will keep both outlers
#Remove rows where life expectency is outside a valid range that could skew our analysis
data= data[(data['Life_expectancy']>50) & (data['Life_expectancy']<100)]

#step6: encode categrical data 
#Method1: one-hot encoding
data_encoded=pd.get_dummies(data, columns=['Prefecture'], drop_first=True)
#convert the boolean output to binary
data_encoded=data_encoded.astype(int)
print(data_encoded.head(50))
#Method2: Label encoding
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data['Prefecture_encoded']=encoder.fit_transform(data['Prefecture'])
print(data.head())

#step7: normalize or standardize or Log_transform numerical data: Scaling
#Standardization
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
#lets apply standardization to 'Life_expectancy' column
#data[['Life_expectancy']]=scaler.fit_transform(data[['Life_expectancy']])
print(data.head(20))
print(data.describe())
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Lets apply normalization on 'Physician_100kP' column
#data[['Physician_100kP']]=scaler.fit_transform(data[['Physician_100kP']])
print(data.head(20))
print(data.describe())
#Log_transform on salary column 
data['Salary_Log']=np.log(data['Salary'])
data=data.drop(columns=['Salary'])
data['Park_Land_%_Log']=np.log(data['Park_Land_%'])
data=data.drop(columns=['Park_Land_%'])
#step9: save the cleaned dataset
data.to_excel('/Users/khalilmosbah/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_log_transform.xlsx', index=False)

#Feature engineering 
import seaborn as sns 
#Select the key features and target
key_features_Strong= ['Life_expectancy', 'Junior_col_%', 'University_%', 'Salary_Log']
key_features_Weak=['Life_expectancy', 'Physician_100kP', 'Park_Land_%_Log']
reverse_key_features=['Life_expectancy','Elementary_school']
#Pair plot
sns.pairplot(data[key_features_Strong], diag_kind="kde", kind="scatter")
plt.title("Pair plot of strong key features and target")
plt.show()
sns.pairplot(data[key_features_Weak], diag_kind="kde", kind="scatter")
plt.title("Pair plot of weak key features and target")
plt.show()
sns.pairplot(data[reverse_key_features], diag_kind="kde", kind="scatter")
plt.title("Pair plot of reverse key feature and target")
plt.show()
#Data Splitting
from sklearn.model_selection import train_test_split
#Split into training and testing sets with X being 2D and y being 1D
X=data[['Junior_col_%', 'University_%', 'Salary_Log', 'Physician_100kP', 'Park_Land_%_Log', 'Elementary_school']]
y=data['Life_expectancy']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
#save the training sets to excel
X_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_log.xlsx', index=False)
y_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_trainy.xlsx', index=False)
X_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_log.xlsx', index=False)
y_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/y_test.xlsx', index=False)
#save training datasets post VIF
X=data[['University_%', 'Junior_col_%']]
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_v2.xlsx', index=False)
X_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_v2.xlsx', index=False)
#RFE Data
X=data.drop(['Life_expectancy','Prefecture'], axis=1)
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
X_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_trainy_RFE.xlsx', index=False)
X_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/X_test_RFE.xlsx', index=False)

