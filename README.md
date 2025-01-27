# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Load the California housing dataset
california_data = fetch_california_housing()

# Convert it into a DataFrame
df = pd.DataFrame(california_data.data, columns=california_data.feature_names)
df['Price'] = california_data.target

# Data exploration and visualization
print(df.head())  # Display first few rows of the dataset

# Visualizing the correlation between features and target variable (Price)
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Splitting the data into features and target
X = df.drop('Price', axis=1)  # Features
y = df['Price']  # Target variable (Price)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for better performance in Ridge regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)

# Predict using the linear regression model
y_pred_linear = linear_reg.predict(X_test_scaled)

# Evaluate the linear regression model
mse_linear = mean_squared_error(y_test, y_pred_linear)
rmse_linear = np.sqrt(mse_linear)
r2_linear = r2_score(y_test, y_pred_linear)

print(f"Linear Regression Model Evaluation:")
print(f"Mean Squared Error: {mse_linear}")
print(f"Root Mean Squared Error: {rmse_linear}")
print(f"R-squared: {r2_linear}\n")

# Ridge Regression Model
ridge_reg = Ridge(alpha=1.0)  # You can adjust alpha for regularization
ridge_reg.fit(X_train_scaled, y_train)

# Predict using the Ridge regression model
y_pred_ridge = ridge_reg.predict(X_test_scaled)

# Evaluate the ridge regression model
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Ridge Regression Model Evaluation:")
print(f"Mean Squared Error: {mse_ridge}")
print(f"Root Mean Squared Error: {rmse_ridge}")
print(f"R-squared: {r2_ridge}\n")

# Plotting the predictions vs actual values for both models
plt.figure(figsize=(10, 6))

# Plot Linear Regression predictions
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_linear, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', color='red')
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

# Plot Ridge Regression predictions
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_ridge, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--k', color='red')
plt.title("Ridge Regression: Actual vs Predicted")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")

plt.tight_layout()
plt.show()
