import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import classification_report
import warnings

# Assuming you have your data loaded into X (features) and y (target variable)

dtype={'TX_AMOUNT': float}
df = pd.read_csv('data/enriched.csv', dtype=dtype)
df_test = pd.read_csv('data/enriched_test.csv', dtype=dtype)


id_columns = ['TX_ID', 'TX_TS', 'CUSTOMER_ID', 'TERMINAL_ID', 'MERCHANT_ID', 'ACQUIRER_ID', 'CARD_EXPIRY_DATE', 'CARD_DATA', 'CARD_BRAND', 'LEGAL_NAME']

# Set the correlation threshold
correlation_threshold = 0.002

X = df.drop(columns=id_columns)
X_final = df_test.drop(columns=id_columns)
y = df['TX_FRAUD']

categorical_columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

for column in categorical_columns:
    codes, uniques = pd.factorize(pd.concat([X[column], X_final[column]]))
    X[column] = codes[:len(X)] + 1
    X_final[column] = codes[len(X):] + 1

X_final = pd.get_dummies(X_final)
X = pd.get_dummies(X)
X, X_final = X.align(X_final, join="left", axis=1)

def map(num):
    if num < 1500:
        return 1
    elif num < 3000:
        return 2
    elif num < 4800:
        return 3
    elif num < 5600:
        return 4
    elif num < 7300:
        return 5
    elif num < 8000:
        return 6
    elif num < 9000:
        return 7
    else:
        return 8

def reg_to_class(num):
    if (num >= 0.5):
        return 1
    else:
        return 0

X['MCC_CODE'] = X['MCC_CODE'].apply(map)

# Calculate the correlation between features and 'TX_FRAUD'
correlation_matrix = X.corr()
correlation_with_target = correlation_matrix['TX_FRAUD']

# Filter features based on the correlation threshold
selected_features = correlation_with_target[correlation_with_target.abs() >= correlation_threshold].index

print(selected_features)

X = X[selected_features].drop(columns='TX_FRAUD')
X_final = X_final[selected_features].drop(columns='TX_FRAUD')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the regression models and their respective parameter grids for tuning
models = {
    # 'K Nearest Neighbor Regressor': (KNeighborsRegressor(), {'n_neighbors': [3, 5, 7]}),
    # 'XGBoost Regressor': (XGBRegressor(), {'n_estimators': [50, 100, 200]}),
    # 'Decision Tree Regressor': (DecisionTreeRegressor(), {'max_depth': [None, 5, 10]}),
    'Support Vector Machine Regressor': (SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}),
    'Random Forest Regressor': (RandomForestRegressor(), {'n_estimators': [50, 100, 200]}),
}

# Perform k-fold cross-validation and parameter tuning for each regression model
for model_name, (model, param_grid) in models.items():
  
  with warnings.catch_warnings():
      warnings.simplefilter(action='ignore', category=FutureWarning)

      grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
      grid_search.fit(X_train, y_train)
      
      best_params = grid_search.best_params_
      best_model = grid_search.best_estimator_
      
      # Evaluate the model using cross-validation with Mean Squared Error (MSE)
      mse_scores = -cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
      mean_mse = np.mean(mse_scores)
      std_mse = np.std(mse_scores)

      # Test the best model on the test set and calculate Root Mean Squared Error (RMSE)
      y_pred = best_model.predict(X_test)
      y_final = best_model.predict(X_final)

      output = pd.DataFrame(y_final, columns=['TX_FRAUD'])
      output['TX_ID'] = df_test['TX_ID']

      output.to_csv(f'output/{model_name}_output.csv', index=False)

      rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
      
      # Calculate additional classification metrics
      binary_pred = [1 if x > 0.5 else 0 for x in y_pred]
      report = classification_report(y_test, binary_pred)
      
      # Test the best model on the test set
      test_accuracy = best_model.score(X_test, binary_pred)
      
      print(f"Model: {model_name}")
      print(f"Best Parameters: {best_params}")
      print(f"Test Set Accuracy: {test_accuracy:.2f}")
      print(f"Cross-Validation Mean MSE: {mean_mse:.2f} (± {std_mse:.2f})")
      print(f"Test Set RMSE: {rmse:.2f}")
      print("Classification Report:")
      print(report)
      print("--------------------------------------------------")
