import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FeatureImportanceAnalysis:
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

    def feature_importance(self, X_train, X_test, y_train, y_test):
        # Train a Random Forest model to evaluate feature importance
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(X_train.shape[1]), importances[indices], align="center")
        plt.yticks(range(X_train.shape[1]), self.X.columns[indices])
        plt.xlabel("Relative Importance")
        plt.show()

        # Display the feature importance
        for f in range(X_train.shape[1]):
            print(f"{self.X.columns[indices[f]]}: {importances[indices[f]]:.4f}")

    def run_analysis(self):
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        self.feature_importance(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    # Set the correct path for the dataset
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")  # Adjust the path if necessary
    data_file = os.path.join(data_folder, 'cleaned_ml.csv')

    # Define the target column for analysis (TotalPremium)
    target_column = 'TotalPremium'

    # Initialize the FeatureImportanceAnalysis class with the data path and target column
    feature_analyzer = FeatureImportanceAnalysis(data_file, target_column)

    # Run the feature importance analysis
    feature_analyzer.run_analysis()