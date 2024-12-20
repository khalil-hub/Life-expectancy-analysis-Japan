import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Step 1: Read the Excel file into a Pandas DataFrame
file_path = '/Users/khalilmosbah/Library/CloudStorage/OneDrive-国立大学法人東海国立大学機構/Weekly challenges_/Data science and Analytics/Japan Life Expectency/Japan_life_expectancy.xlsx'
df = pd.read_excel(file_path)

# Step 2: Select and encode columns
x = df['Prefecture']  # This is the data for the x-axis
y = df['Life_expectancy']  # This is the data for the y-axis
z = df['Physician']

# Label encoding for Prefecture
label_encoder = LabelEncoder()
df['Prefecture_encoded'] = label_encoder.fit_transform(df['Prefecture'])

# Step 3: Sort the data by 'Life_expectancy'
df_sorted = df.sort_values('Life_expectancy')

# Step 4: Plot life expectancy as a line plot
plt.figure(figsize=(14, 10))
plt.plot(df_sorted['Prefecture'], df_sorted['Life_expectancy'], label='Life Expectancy', color='b', linestyle='-', marker='o')
plt.title('Life Expectancy per Prefecture')
plt.xlabel('Prefecture')
plt.ylabel('Life Expectancy')
plt.xticks(fontsize=8, fontweight='bold', rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# Graph 2: Physician per Prefecture (Regression)
X = np.array(df_sorted['Prefecture_encoded']).reshape(-1, 1)  # Reshape to 2D array
y = df_sorted['Physician']

# Linear regression
model = LinearRegression()
model.fit(X, y)
z_pred = model.predict(X)

# Plotting regression results
fig, ax = plt.subplots(figsize=(14, 10))
ax.scatter(df_sorted['Prefecture_encoded'], df_sorted['Physician'], label='Physician', color='b')
ax.plot(df_sorted['Prefecture_encoded'], z_pred, color="r", label="Regression Line")
ax.set_title('Physician per Prefecture', fontsize=14)
ax.set_xlabel('Encoded Prefecture', fontsize=12)
ax.set_ylabel('Physician per 100k people', fontsize=12)
# Set x-axis to show prefecture names
ax.set_xticks(df_sorted['Prefecture_encoded'])  # Use encoded values for tick positions
ax.set_xticklabels(df_sorted['Prefecture'], rotation=45, fontsize=9, fontweight='bold')  # Use prefecture names for tick labels

ax.legend()
plt.tight_layout()
plt.show()
