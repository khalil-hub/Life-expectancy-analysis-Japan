import pandas as pd
# Understanding, cleaning and preprocessing the Data (Check Data_Pre_Processing.py file)

#Load the data
#Load the Raw excel file using the directory of the file and read the file using pandas library excel reading function
raw_data_path='~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/raw/Japan_Life_Expectancy.xlsx'
data=pd.read_excel(raw_data_path)

#Rename columns for clarity
#rename columns
data.rename(columns={'Physician': 'Physician_100kP', 'Junior_col': 'Junior_col_%', 'University': 'University_%', 'Public_Hosp': 'Public_Hosp_%', 'Pshic_hosp': 'Psych_Hosp_100kP', 'Beds_psic': 'Psych_Beds_100kP', 'Nurses': 'Nurses_100kP', 'Avg_hours': 'Avg_Work_Hours_Month', 'Elementary_School': 'Elementary_School_%', 'Sport_fac': 'Sport_fac_1MP', 'Park': 'Park_Land_%', 'Forest': 'Forest_Land_%', 'Income_per capita': 'Income_capita', 'Density_pop': 'Population_Density_km2', 'Hospitals': 'General_Hospital_100kP', 'Beds': 'General_Hospital_Beds_100k', 'Ambulances': 'Ambulances_100kP', 'Health_exp': 'Health_Expenditure_%', 'Educ_exp': 'Educ_Expenditure_%', 'Welfare_exp': 'Welfare_Expenditure_%'}, inplace=True)
print(data.info())

#Encode categrical data 
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

#normalize or standardize or Log_transform numerical data: Scaling
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
numerical_features=data.select_dtypes(include=['float64', 'int64']).columns #include only numerical values
numerical_features=numerical_features.drop(['Prefecture_encoded', 'Life_expectancy'])
data[numerical_features]=scaler.fit_transform(data[numerical_features])
#save the cleaned dataset
data.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy.xlsx', index=False)

#Feature engineering 
import seaborn as sns 
#Select the key features and target
key_features=['Life_expectancy', 'Physician_100kP','Junior_col_%', 'University_%', 'Salary','Elementary_school', 'Park_Land_%', 'Ambulances_100kP']
#Data Splitting
from sklearn.model_selection import train_test_split

#Split into training and testing sets with X being 2D and y being 1D
X=data[['Physician_100kP','Junior_col_%', 'University_%', 'Salary','Elementary_school', 'Park_Land_%', 'Ambulances_100kP']]
y=data['Life_expectancy']
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape)
#save the training sets to excel
X_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_train.xlsx', index=False)
X_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_X_test.xlsx', index=False)
y_train.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_train.xlsx', index=False)
y_test.to_excel('~/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy_y_test.xlsx', index=False)
