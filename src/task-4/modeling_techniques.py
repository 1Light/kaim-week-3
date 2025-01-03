import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

class ModelingTechniques:
    def __init__(self, data):
        # Assume data is pre-processed and passed as argument
        self.data = data
        # Identify non-numeric columns
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting non-numeric columns: {non_numeric_cols}")
            # Convert date columns to datetime and extract features if necessary
            for col in non_numeric_cols:
                if pd.to_datetime(self.data[col], errors='coerce').notna().all():
                    self.data[col] = pd.to_datetime(self.data[col]).apply(lambda x: x.toordinal())
                else:
                    # Handle categorical data
                    self.data[col] = pd.factorize(self.data[col])[0]

        # Drop 'TotalClaims' since it's empty and not used
        self.X = self.data.drop(['TotalPremium'], axis=1)  # Drop only 'TotalPremium' from features
        self.y = self.data['TotalPremium']  # Target: 'TotalPremium'

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

    def decision_tree(self, X_train, X_test, y_train, y_test):
        print("Training Decision Tree Model...")
        model = DecisionTreeRegressor(random_state=42)
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
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
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

    def run_all_models(self):
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        
        self.linear_regression(X_train, X_test, y_train, y_test)
        self.decision_tree(X_train, X_test, y_train, y_test)
        self.random_forest(X_train, X_test, y_train, y_test)
        self.xgboost(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    # Set the correct path for the dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")  # Update with your actual folder structure
    data_file = os.path.join(data_folder, 'cleaned_ml.csv')

    # Load the data into a pandas DataFrame
    data = pd.read_csv(data_file, low_memory=False)

    # Initialize the ModelingTechniques class with the data
    modeler = ModelingTechniques(data)

    # Run all models and evaluate performance
    modeler.run_all_models()