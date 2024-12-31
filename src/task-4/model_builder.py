import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

class ModelBuilder:
    def __init__(self, data_path):
        # Load and preprocess the data
        self.data = pd.read_csv(data_path, low_memory=False)
        self.preprocess_data()

    def preprocess_data(self):
        # Identify and convert non-numeric columns
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting non-numeric columns: {non_numeric_cols}")
            for col in non_numeric_cols:
                if pd.to_datetime(self.data[col], errors='coerce').notna().all():
                    # Convert date columns to ordinal values
                    self.data[col] = pd.to_datetime(self.data[col]).apply(lambda x: x.toordinal())
                else:
                    # Factorize categorical columns
                    self.data[col] = pd.factorize(self.data[col])[0]

        # Handle missing values (if necessary)
        self.data.fillna(self.data.mean(), inplace=True)  # Replace missing numeric values with the mean
        self.data.fillna(0, inplace=True)  # Replace missing categorical values with 0 (or use other strategies)

        # Split features (X) and target variable (y)
        self.X = self.data.drop(['TotalPremium'], axis=1) 
        self.y = self.data['TotalPremium'] 

    def train_test_split_data(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def linear_regression(self, X_train, X_test, y_train, y_test):
        print("Training Linear Regression Model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_model(y_test, y_pred, model)

    def random_forest(self, X_train, X_test, y_train, y_test):
        print("Training Random Forest Model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_model(y_test, y_pred, model)

    def xgboost(self, X_train, X_test, y_train, y_test):
        print("Training XGBoost Model...")
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_model(y_test, y_pred, model)

    def evaluate_model(self, y_test, y_pred, model):
        # Calculate performance metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Model: {model.__class__.__name__}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("-" * 50)

    def run_models(self):
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        self.linear_regression(X_train, X_test, y_train, y_test)
        self.random_forest(X_train, X_test, y_train, y_test)
        self.xgboost(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # Set the correct path for the dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")  # Update with your actual folder structure
    data_file = os.path.join(data_folder, 'cleaned_ml.csv')  # Ensure this is the correct file

    # Initialize the ModelBuilder class with the data path
    model_builder = ModelBuilder(data_file)

    # Run models and evaluate performance
    model_builder.run_models()