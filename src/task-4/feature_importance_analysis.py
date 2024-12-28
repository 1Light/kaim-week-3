import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureImportanceAnalysis:
    def __init__(self, data_path, results_dir):
        """
        Initializes the FeatureImportanceAnalysis object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for feature importance results
        self.feature_importance_dir = os.path.join(self.results_dir, "feature_importance_results")
        os.makedirs(self.feature_importance_dir, exist_ok=True)

    def plot_feature_importance(self, model, model_name, feature_names):
        """
        Plots and saves the feature importance for a given model.

        :param model: The trained model.
        :param model_name: The name of the model (Random Forest or XGBoost).
        :param feature_names: List of feature names.
        """
        # Get feature importances
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_booster().get_score(importance_type='weight')
        importances = list(importances.values()) if isinstance(importances, dict) else importances

        # Create a DataFrame for feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort the DataFrame by importance
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        # Plot the feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"Feature Importance - {model_name}")
        plt.tight_layout()

        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = os.path.join(self.feature_importance_dir, f"feature_importance_{model_name}_{timestamp}.png")
        plt.savefig(plot_file)
        print(f"Feature importance plot saved as: {plot_file}")

    def analyze_feature_importance(self):
        """
        Analyzes feature importance using Random Forest and XGBoost models.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])  # Features
        y = self.data['TotalClaims']  # Target variable (or use 'TotalPremium' if applicable)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to analyze feature importance
        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor()
        }

        # Evaluate each model and plot feature importance
        for model_name, model in models.items():
            print(f"Training {model_name} model and analyzing feature importance...")
            model.fit(X_train, y_train)
            self.plot_feature_importance(model, model_name, X.columns)

# Example usage of the FeatureImportanceAnalysis class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the FeatureImportanceAnalysis object
    feature_importance_analysis = FeatureImportanceAnalysis(data_path=cleaned_csv_file, results_dir=results_dir)

    # Analyze feature importance
    feature_importance_analysis.analyze_feature_importance()