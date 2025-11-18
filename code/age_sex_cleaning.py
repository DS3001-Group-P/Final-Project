import pandas as pd
import numpy as np

data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

columns_of_interest = ['Age', 'Sex', 'Systolic BP', 'Diastolic BP']
data_cleaned = data[columns_of_interest].copy()

data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})
data_cleaned = data_cleaned.dropna()

data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex'], drop_first=True)

# Convert Age to int64
data_cleaned['Age'] = pd.to_numeric(data_cleaned['Age'], errors='coerce').astype('int64')

# Convert encoded Sex column to int64
data_cleaned['Sex_M'] = data_cleaned['Sex_M'].astype('int64')

data_cleaned.to_csv('/workspaces/Final-Project/code/CVD_Cleaned.csv', index=False)

