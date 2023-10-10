import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Read the data
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input", "Payments Fraud Dataset")
dtype = {"IS_RECURRING_TRANSACTION": str}
X_full = pd.read_csv(os.path.join(path, "transactions_train.csv"), dtype=dtype)
X_test_full = pd.read_csv(os.path.join(path, "transactions_test.csv"), dtype=dtype)
X_full.IS_RECURRING_TRANSACTION.replace("Fals", "False", inplace=True)
X_test_full.IS_RECURRING_TRANSACTION.replace("Fals", "False", inplace=True)

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=["TX_FRAUD"], inplace=True)
y = X_full.TX_FRAUD
X_full.drop(["TX_FRAUD"], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
X_train, X_test = X_train.align(X_test, join="left", axis=1)

# Define the model
model = XGBRegressor(n_estimators=1000, early_stopping_rounds=5)

# Fit the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

# Get predictions
predictions = model.predict(X_valid)

# Calculate MAE
mae = mean_absolute_error(predictions, y_valid)
print("Mean Absolute Error:", mae)
