import matplotlib
matplotlib.use('Age')  # Use a non-GUI backend to avoid font issues

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("heart.csv")  # Ensure heart.csv is in the same folder as your script
print(df.head())

# Explore the dataset
print(df.info())
print(df.describe())

# --- FIX: Clean column names and target values ---
# Remove extra spaces in column names
df.columns = df.columns.str.strip()

# Ensure column is renamed to 'target'
if "Heart Disease" in df.columns:
    df.rename(columns={"Heart Disease": "target"}, inplace=True)

# Clean trailing spaces in target values
df["target"] = df["target"].str.strip()

# Map Presence/Absence -> 1/0
df["target"] = df["target"].map({"Presence": 1, "Absence": 0})

print("Target value counts:\n", df["target"].value_counts())

# Visualize target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title('Target Distribution (0 = No Disease, 1 = Disease)')
plt.savefig("target_distribution.png")
plt.clf()  # Clear figure for next plot

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Separate features and target
X = df.drop('target', axis=1)
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Accuracy & classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix heatmap
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig("confusion_matrix.png")
plt.clf()

# Make a sample prediction
sample = X_test.iloc[0].values.reshape(1, -1)
predicted = model.predict(sample)
actual = y_test.iloc[0]
print(f"Sample Prediction: {predicted[0]} | Actual: {actual}")

# Feature importance
importances = model.feature_importances_
features = X.columns

fi_df = pd.DataFrame({'Feature': features, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=fi_df)
plt.title('Feature Importance')
plt.savefig("feature_importance.png")
plt.clf()
