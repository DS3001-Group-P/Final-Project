import pandas as pd
import numpy as np

data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

columns_of_interest = ['Age', 'Sex', 'Systolic BP', 'Diastolic BP']
data_cleaned = data[columns_of_interest].copy()
data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})

# Drop rows where Age is NaN
data_cleaned.dropna(subset=['Age'], inplace=True)
# Convert Age to int64
data_cleaned['Age'] = data_cleaned['Age'].astype('int64')

# Drop rows with NaN values in BP
data_cleaned.dropna(subset=['Systolic BP', 'Diastolic BP'], inplace=True)
# BP to int64
data_cleaned['Systolic BP'] = data_cleaned['Systolic BP'].astype('int64')
data_cleaned['Diastolic BP'] = data_cleaned['Diastolic BP'].astype('int64')

def get_blood_pressure_category(systolic_bp, diastolic_bp):
    
    if systolic_bp < 120 and diastolic_bp < 80:
        return 'Normal'
    elif 120 <= systolic_bp <= 129 and diastolic_bp < 80:
        return 'Elevated'
    elif (130 <= systolic_bp <= 139) or (80 <= diastolic_bp <= 89):
        return 'High (Stage 1)'
    elif (systolic_bp >= 140) or (diastolic_bp >= 90):
        return 'High (Stage 2)'
    elif (systolic_bp > 180) or (diastolic_bp > 120):
        return 'Hypertensive Crisis'
    else:
        return 'Undefined'

data_cleaned['Blood Pressure Category'] = data_cleaned.apply(lambda row: get_blood_pressure_category(row['Systolic BP'], row['Diastolic BP']), axis=1)

data_cleaned.to_csv('CVD_Cleaned.csv', index=False)

print(data_cleaned.head(20))