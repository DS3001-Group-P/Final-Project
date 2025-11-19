import pandas as pd
import numpy as np

data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

# Clean
columns_of_interest = ['Age', 'Sex', 'Blood Pressure Category']
data_cleaned = data[columns_of_interest].copy()
data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})

# Drop rows where Age or Blood Pressure Category are NaN
data_cleaned.dropna(subset=['Age', 'Blood Pressure Category'], inplace=True)

# Convert Age to int64
data_cleaned['Age'] = data_cleaned['Age'].astype('int64')

# ________________________________________________________________________________


# Save cleaned dataframe to CSV
data_cleaned.to_csv('CVD_Cleaned.csv', index=False)

# Encode Sex as numeric
data_cleaned['Sex'] = data_cleaned['Sex'].map({'F': 0, 'M': 1})

# Encode Blood Pressure Category as numeric
bp_map = {
    'Normal': 0,
    'Elevated': 1,
    'Hypertension Stage 1': 2,
    'Hypertension Stage 2': 3
}
data_cleaned['BP_Category_Label'] = data_cleaned['Blood Pressure Category'].map(bp_map)

# Display
print(data_cleaned.dtypes)
print(data_cleaned.describe())
print(data_cleaned.head(10))

model_data = data_cleaned.copy()
model_data.to_csv('model_data.csv', index=False)
