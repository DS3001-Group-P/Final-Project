import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


data_cleaned = pd.read_csv('/workspaces/Final-Project/code/CVD_Cleaned.csv')

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

print(data_cleaned.head())