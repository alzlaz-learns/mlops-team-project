# src/train_model.py

import os

import joblib
import pandas as pd
from scipy.io import arff
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ?? Full path to your dataset
dataset_path = r"C:\Users\aleks\mlops-team-project\data\raw\diabetes.arff"

# ?? Full path to save model
model_output_path = r"C:\Users\aleks\mlops-team-project\models\random_forest_diabetes.pkl"

# Step 1: Load ARFF data
data, meta = arff.loadarff(dataset_path)
df = pd.DataFrame(data)

# Step 2: Decode bytes (if needed)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].map(lambda x: x.decode() if isinstance(x, bytes) else x)

# ?? DEBUG: Print column names
print("?? Column names in dataset:", df.columns)

# ? Fixed target column
target_column = "Outcome"

# Step 3: Prepare features and labels
if target_column not in df.columns:
    raise ValueError(f"? ERROR: Target column '{target_column}' not found in dataset. Please update the script.")
X = df.drop(target_column, axis=1)
y = df[target_column]

# Step 4: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Step 5: Train model
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# Step 6: Evaluate
y_pred = rf.predict(X_test)
print("? Accuracy:", accuracy_score(y_test, y_pred))
print("?? Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
joblib.dump(rf, model_output_path)
print(f"?? Model saved to: {model_output_path}")
