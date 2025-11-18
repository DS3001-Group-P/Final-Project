import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

# Keep only Age and Sex
columns_of_interest = ['Age', 'Sex']
data_cleaned = data[columns_of_interest].copy()

# Replace missing values with NaN
data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})
data_cleaned = data_cleaned.fillna(np.nan)

# One-hot encode Sex column
data_cleaned = pd.get_dummies(data_cleaned, columns=['Sex'], drop_first=True)

# Save cleaned data
data_cleaned.to_csv('CVD_Cleaned.csv', index=False)

