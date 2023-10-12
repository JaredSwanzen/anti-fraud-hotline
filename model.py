import os
import pytz
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
terminals = pd.read_csv(os.path.join(path, "terminals.csv"))
customers = pd.read_csv(os.path.join(path, "customers.csv"))
merchants = pd.read_csv(os.path.join(path, "merchants.csv"))

# Join tables
X_full = X_full.join(terminals.set_index("TERMINAL_ID"), "TERMINAL_ID", "left")
X_full = X_full.join(customers.set_index("CUSTOMER_ID"), "CUSTOMER_ID", "left")
X_full = X_full.join(merchants.set_index("MERCHANT_ID"), "MERCHANT_ID", "left")
X_test_full = X_test_full.join(terminals.set_index("TERMINAL_ID"), "TERMINAL_ID", "left")
X_test_full = X_test_full.join(customers.set_index("CUSTOMER_ID"), "CUSTOMER_ID", "left")
X_test_full = X_test_full.join(merchants.set_index("MERCHANT_ID"), "MERCHANT_ID", "left")

# Some cleaning
X_full.drop(["TERMINAL_ID"], axis=1, inplace=True)
X_full.drop(["CUSTOMER_ID"], axis=1, inplace=True)
X_full.drop(["MERCHANT_ID"], axis=1, inplace=True)
X_full.IS_RECURRING_TRANSACTION.replace("Fals", "False", inplace=True)
X_full.FAILURE_REASON.fillna("Did not fail", inplace=True)
X_test_full.drop(["TERMINAL_ID"], axis=1, inplace=True)
X_test_full.drop(["CUSTOMER_ID"], axis=1, inplace=True)
X_test_full.drop(["MERCHANT_ID"], axis=1, inplace=True)
X_test_full.IS_RECURRING_TRANSACTION.replace("Fals", "False", inplace=True)
X_test_full.FAILURE_REASON.fillna("Did not fail", inplace=True)
print(X_full.columns)

# Some enriching
X_full["CARD_EXPIRY_DATE"] = pd.to_datetime(X_full.CARD_EXPIRY_DATE, format="%m/%y").dt.tz_localize(pytz.UTC)
X_full["TX_TS"] = pd.to_datetime(X_full.TX_TS)
X_full["TRANSACTION_TO_EXPIRY_DISTANCE"] = (X_full.CARD_EXPIRY_DATE - X_full.TX_TS).dt.days
X_full["CUSTOMER_TO_TERMINAL_DISTANCE"] = np.sqrt(
    (X_full.x_customer_id - X_full.x_terminal_id) ** 2 + (X_full.y_customer_id - X_full.y_terminal__id) ** 2
)
X_full["TRANSACTION_TOTAL"] = X_full.TX_AMOUNT - X_full.TRANSACTION_CASHBACK_AMOUNT

X_test_full["CARD_EXPIRY_DATE"] = pd.to_datetime(X_test_full.CARD_EXPIRY_DATE, format="%m/%y").dt.tz_localize(pytz.UTC)
X_test_full["TX_TS"] = pd.to_datetime(X_test_full.TX_TS)
X_test_full["TRANSACTION_TO_EXPIRY_DISTANCE"] = (X_test_full.CARD_EXPIRY_DATE - X_test_full.TX_TS).dt.days
X_test_full["CUSTOMER_TO_TERMINAL_DISTANCE"] = np.sqrt(
    (X_test_full.x_customer_id - X_test_full.x_terminal_id) ** 2
    + (X_test_full.y_customer_id - X_test_full.y_terminal__id) ** 2
)
X_test_full["TRANSACTION_TOTAL"] = X_test_full.TX_AMOUNT - X_test_full.TRANSACTION_CASHBACK_AMOUNT

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=["TX_FRAUD"], inplace=True)
y = X_full.TX_FRAUD
X_full.drop(["TX_FRAUD"], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2)

# # "Cardinality" means the number of unique values in a column
# # Select categorical columns with relatively low cardinality (convenient but arbitrary)
# low_cardinality_cols = [
#     cname
#     for cname in X_train_full.columns
#     if X_train_full[cname].nunique() < 10 and X_train_full[cname].dtype == "object"
# ]
# print("Low cardinality columns:", low_cardinality_cols)

# # Select numerical columns
# numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ["int64", "float64"]]
# print("Numerical columns:", numerical_cols)

# # Keep selected columns only
# my_cols = (
#     [
#         "FAILURE_REASON",
#         "TRANSACTION_CURRENCY",
#         "CARD_COUNTRY_CODE",
#         "TAX_EXCEMPT_INDICATOR",
#     ]
#     + low_cardinality_cols
#     + numerical_cols
# )
# print("All columns:", my_cols)

my_cols = [
    "TX_AMOUNT",
    "TRANSACTION_GOODS_AND_SERVICES_AMOUNT",
    "TRANSACTION_CASHBACK_AMOUNT",
    "TRANSACTION_TYPE",
    "CARD_COUNTRY_CODE",
    "IS_RECURRING_TRANSACTION",
    "CARDHOLDER_AUTH_METHOD",
    "x_customer_id",
    "y_customer_id",
    "x_terminal_id",
    "y_terminal__id",
    "MCC_CODE",
    "OUTLET_TYPE",
    "DEPOSIT_REQUIRED_PERCENTAGE",
    "DEPOSIT_PERCENTAGE",
    "TRANSACTION_TO_EXPIRY_DISTANCE",
    "TRANSACTION_TOTAL",
]
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data (to shorten the code, we use pandas)
X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_train, X_valid = X_train.align(X_valid, join="left", axis=1)
X_train, X_test = X_train.align(X_test, join="left", axis=1)
print(X_train.columns)

# Define the model
model = XGBRegressor(n_estimators=1000, learning_rate=0.05, early_stopping_rounds=5)

# Fit the model
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)])

# Get predictions
predictions = model.predict(X_valid)

# Calculate MAE
mae = mean_absolute_error(predictions, y_valid)
print("Mean Absolute Error:", mae)

# Get accuracy score
print(model.score(X_valid, y_valid))

# Make predictions on the test data
answers = model.predict(X_test)
output = pd.DataFrame({"TX_ID": X_test_full.TX_ID, "TX_FRAUD": answers})
output.set_index("TX_ID", inplace=True)
print(output.head())
output.to_csv(os.path.join(path, "output.csv"))
