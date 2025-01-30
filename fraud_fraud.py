# Importing Necessary Libraries
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Loading the Dataset
data = pd.read_csv("PS_20174392719_1491204439457_log.csv")
print(data.head())
print(data.describe())
columns_to_keep = ['column1','column2']
data=data[columns_to_keep]
#reduced file
data.to_csv('reduced_file.csv', index=False)

# Checking for missing values and data types
data.info()
print(data.isnull().sum())

# Transaction Type Analysis
type_counts = data["type"].value_counts()
type_transactions = type_counts.index
type_values = type_counts.values

# Bar chart visualization
figure_bar = px.bar(x=type_transactions, y=type_values, color=type_values, color_continuous_scale="sunset")
figure_bar.show()

# Class evaluation
print(data["isFraud"].value_counts())

# Pie chart visualization
figure_pie = px.pie(data, values=type_values, names=type_transactions, title="Types of Transaction")
figure_pie.show()

# Correlation Analysis
correlation = data.select_dtypes(include=[np.number]).corr()
print(correlation["isFraud"].sort_values(ascending=False))
plot = px.imshow(correlation)
plot.show()

# Preprocessing Categorical Variables
data['type'] = data['type'].map({'CASH_OUT': 1, 'PAYMENT': 2, 'CASH_IN': 3, 'TRANSFER': 4, 'DEBIT': 5})

# Features and Target Selection
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data["isFraud"])

# Handling Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
x_resampled, y_resampled = smote.fit_resample(x, y)
print("Class distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

# Feature Scaling with StandardScaler
scaler = StandardScaler()
x_resampled = scaler.fit_transform(x_resampled)

# Train-Test Split
xtrain, xtest, ytrain, ytest = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=42)

# Decision Tree Classifier Model
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)

# Evaluation
print("Model Accuracy:", model.score(xtest, ytest) * 100)

# Example Predictions
example_features_1 = np.array([[4, 9000.60, 9000.60, 0.00]])
example_features_2 = np.array([[2, 9839.64, 170136.00, 160296.36]])

# Scale example features before prediction
example_features_1 = scaler.transform(example_features_1)
example_features_2 = scaler.transform(example_features_2)

# Map predictions to labels
prediction_labels = {1: "Fraud", 0: "No Fraud"}

print("Example 1 Prediction:", prediction_labels[model.predict(example_features_1)[0]])
print("Example 2 Prediction:", prediction_labels[model.predict(example_features_2)[0]])
