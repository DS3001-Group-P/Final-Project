import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


data = pd.read_csv('/workspaces/Final-Project/data/CVD Dataset.csv')

columns_of_interest = ['Age', 'Sex', 'Blood Pressure Category']
data_cleaned = data[columns_of_interest].copy()
data_cleaned = data_cleaned.replace({None: np.nan, '': np.nan})

# Drop rows where Age or Blood Pressure Category are NaN
data_cleaned.dropna(subset=['Age', 'Blood Pressure Category'], inplace=True)

# Convert Age to int64
data_cleaned['Age'] = data_cleaned['Age'].astype('int64')

# Final data info and preview
print(data_cleaned.info())
print(data_cleaned.head())

# Save cleaned dataframe to CSV
data_cleaned.to_csv('CVD_Cleaned.csv', index=False)
# ________________________________________________________________________________



# Models
# Prepare data
X = data_cleaned[['Age', 'Sex']].copy()

# Encode Sex column to numeric values
sex_encoder = LabelEncoder()
X['Sex'] = sex_encoder.fit_transform(X['Sex'])

y = data_cleaned['Blood Pressure Category']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Classification Report:")
print(classification_report(y_test, y_pred_dt, target_names=le.classes_))

plt.figure(figsize=(12,8))
plot_tree(dt, feature_names=X.columns, class_names=le.classes_, filled=True)
plt.show()

# Random forest
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Feature Iportance
importances = rf.feature_importances_
plt.barh(X.columns, importances)
plt.title('Random Forest Feature Importance')
plt.show()