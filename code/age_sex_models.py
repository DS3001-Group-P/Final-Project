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
print(data_cleaned.head(5))