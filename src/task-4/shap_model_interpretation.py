import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

class SHAPModelInterpretation:
    def __init__(self, data_path, target_column):
        # Load the dataset
        self.data = pd.read_csv(data_path, low_memory=False)
        self.target_column = target_column
        self.preprocess_data()

    def preprocess_data(self):
        # Convert non-numeric columns into numeric values
        non_numeric_cols = self.data.select_dtypes(include=['object']).columns
        if len(non_numeric_cols) > 0:
            print(f"Converting non-numeric columns: {non_numeric_cols}")
            for col in non_numeric_cols:
                if pd.to_datetime(self.data[col], errors='coerce').notna().all():
                    self.data[col] = pd.to_datetime(self.data[col]).apply(lambda x: x.toordinal())
                else:
                    self.data[col] = pd.factorize(self.data[col])[0]

        # Split features and target variables
        self.X = self.data.drop(self.target_column, axis=1)  # All columns except target
        self.y = self.data[self.target_column]  # The target column

    def train_test_split_data(self):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        # Train a Random Forest model for regression
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model

    def interpret_with_shap(self, model, X_train):
        # Create a SHAP explainer for the model
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)

        # Plot the SHAP values for the features
        shap.summary_plot(shap_values, X_train)

        # Display the SHAP values for the first instance
        shap.initjs()
        shap.force_plot(explainer.expected_value[0], shap_values[0], X_train.iloc[0])

    def run_analysis(self):
        # Train-test split
        X_train, X_test, y_train, y_test = self.train_test_split_data()

        # Train the model
        model = self.train_model(X_train, y_train)

        # Interpret the model with SHAP
        self.interpret_with_shap(model, X_train)

if __name__ == "__main__":
    # Set the correct path for the dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")  # Adjust the path if necessary
    data_file = os.path.join(data_folder, 'cleaned_ml.csv')

    # Define the target column for analysis (TotalPremium)
    target_column = 'TotalPremium'

    # Initialize the SHAPModelInterpretation class with the data path and target column
    shap_interpreter = SHAPModelInterpretation(data_file, target_column)

    # Run the SHAP interpretation analysis
    shap_interpreter.run_analysis()