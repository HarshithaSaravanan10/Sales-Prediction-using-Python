# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\Harshitha\Downloads\Advertising.csv"
df = pd.read_csv(file_path)

# Display the first few rows
print("Dataset Preview:\n", df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Check dataset information
print("\nDataset Info:")
print(df.info())

# Rename columns if needed (Check column names)
print("\nColumn Names:", df.columns)

# Drop any unnecessary columns (e.g., 'Unnamed' columns if present)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Display basic statistics
print("\nDataset Description:\n", df.describe())

### **🔹 Step 2: Visualizing the Data**

# Pairplot to check relationships between variables
sns.pairplot(df)
plt.show()

# Heatmap to check correlation
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

### **🔹 Step 3: Data Preprocessing**

# Define features (X) and target variable (y)
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables
y = df['Sales']  # Dependent variable

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining Set Size: {X_train.shape}, Testing Set Size: {X_test.shape}")

### **🔹 Step 4: Train the Linear Regression Model**

# Create the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Get model coefficients
print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

### **🔹 Step 5: Make Predictions and Evaluate the Model**

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display evaluation metrics
print("\nModel Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

### **🔹 Step 6: Visualizing Predictions vs. Actual Sales**

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, color="blue", alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color="red", linestyle="--")  # 45-degree line
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.show()
