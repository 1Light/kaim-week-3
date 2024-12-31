import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

class ModelEvaluation:
    def __init__(self, data_path, target_column):
        # Load and preprocess the data
        self.data = pd.read_csv(data_path, low_memory=False)
        self.target_column = target_column
        self.preprocess_data()

    def preprocess_data(self):
        # Identify and convert non-numeric columns
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting non-numeric columns: {non_numeric_cols}")
            for col in non_numeric_cols:
                if pd.to_datetime(self.data[col], errors='coerce').notna().all():
                    self.data[col] = pd.to_datetime(self.data[col]).apply(lambda x: x.toordinal())
                else:
                    self.data[col] = pd.factorize(self.data[col])[0]

        # Split features and target variables
        self.X = self.data.drop(self.target_column, axis=1)  # Features
        self.y = self.data[self.target_column]  # Target

    def train_test_split_data(self):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def linear_regression(self, X_train, X_test, y_train, y_test):
        print("Evaluating Linear Regression Model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_metrics(y_test, y_pred, model)

    def random_forest(self, X_train, X_test, y_train, y_test):
        print("Evaluating Random Forest Regressor...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_metrics(y_test, y_pred, model)

    def xgboost(self, X_train, X_test, y_train, y_test):
        print("Evaluating XGBoost Regressor...")
        model = XGBRegressor(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        self.evaluate_metrics(y_test, y_pred, model)

    def evaluate_metrics(self, y_test, y_pred, model):
        # Calculate performance metrics for regression
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model: {model.__class__.__name__}")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"R-squared: {r2:.4f}")
        print("-" * 50)

    def run_evaluation(self):
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        self.linear_regression(X_train, X_test, y_train, y_test)
        self.random_forest(X_train, X_test, y_train, y_test)
        self.xgboost(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # Set the correct path for the dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")  
    data_file = os.path.join(data_folder, 'cleaned_ml.csv')

    # Replace 'TotalPremium' with your actual target column if needed
    target_column = 'TotalPremium'

    # Initialize the ModelEvaluation class with the data path and target column
    model_evaluator = ModelEvaluation(data_file, target_column)

    # Run evaluations and print metrics
    model_evaluator.run_evaluation()