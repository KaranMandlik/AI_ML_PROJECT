import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Provide the correct full path to the CSV file
file_path = 'C:/Documents/house_price.csv'

# Load dataset with error handling
try:
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print(data.head())  # Print the first few rows of the dataset to verify
except FileNotFoundError:
    print(f"The file at {file_path} was not found. Please check the file path.")
    exit()
except pd.errors.EmptyDataError:
    print("The file is empty. Please provide a valid dataset.")
    exit()

# Perform basic EDA
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Features and target variable
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Optionally scale the features (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Debugging prints before splitting data
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Optionally, print the model's coefficients and intercept
print(f"Model Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Visualizing the relationship between 'area' and 'price'
plt.scatter(data['area'], data['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('House Price vs Area')
plt.show(block=True)  # Keep the plot window open until closed manually
