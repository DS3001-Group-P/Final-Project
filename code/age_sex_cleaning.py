import pandas as pd
import numpy as np

data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

columns_of_interest = ['Age', 'Sex', 'Blood Pressure Category']
data_cleaned = data[columns_of_interest].copy()
data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})

# Drop rows where Age or Blood Pressure Category are NaN
data_cleaned.dropna(subset=['Age', 'Blood Pressure Category'], inplace=True)

# Convert Age to int64
data_cleaned['Age'] = data_cleaned['Age'].astype('int64')

# Final data
print(data_cleaned.info())
print(data_cleaned.head())

# Save cleaned dataframe to CSV
data_cleaned.to_csv('CVD_Cleaned.csv', index=False)
