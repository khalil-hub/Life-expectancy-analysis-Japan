import pandas as pd

#Step1: Load the data
#Load the Raw excel file using the directory of the file and read the file using pandas library excel reading function
raw_data_path='/Users/khalilmosbah/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/raw/Japan_Life_Expectancy.xlsx'
data=pd.read_excel(raw_data_path)

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
data.rename(columns={'Physician': 'Physician_100kP', 'Junior_col': 'Junior_col_%', 'University': 'University_%', 'Public_Hosp': 'Public_Hosp_%', 'Pshic_hosp': 'Psych_Hosp_100kP', 'Beds_psic': 'Psych_Beds_100kP', 'Nurses': 'Nurses_100kP', 'Avg_hours': 'Avg_Work_Hours_Month', 'Elementary_School': 'Elementary_School_%', 'Sport_fac': 'Sport_fac_1MP', 'Park': 'Park_Land_%', 'Forest': 'Forest_Land_%', 'Income_per capita': 'Income_Person', 'Density_pop': 'Population_Density_km2', 'Hospitals': 'General_Hospital_100kP', 'Beds': 'General_Hospital_Beds_100k', 'Ambulances': 'Ambulances_100kP', 'Health_exp': 'Health_Expenditure_%', 'Educ_exp': 'Educ_Expenditure_%', 'Welfare_exp': 'Welfare_Expenditure_%'}, inplace=True)
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

#step7: normalize or standardize numerical data: Scaling
#Standardization
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
#lets apply standardization to 'Life_expectancy' column
data[['Life_expectancy']]=scaler.fit_transform(data[['Life_expectancy']])
print(data.head(20))
print(data.describe())
#Normalization
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
#Lets apply normalization on 'Physician_100kP' column
data[['Physician_100kP']]=scaler.fit_transform(data[['Physician_100kP']])
print(data.head(20))
print(data.describe())

#step8: split data into features and target 
X=data['Income_Person']
Y=data['Life_expectancy']
plt.plot(X, Y)
plt.show()

#step9: save the cleaned dataset
data.to_excel('/Users/khalilmosbah/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly_challenges/Data science and Analytics/Japan_Life_Expectency/data/processed/Cleaned_Japan_Life_Expectancy.xlsx', index=False)