import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

class ModelInterpretationSHAP:
    def __init__(self, data_path, results_dir):
        """
        Initializes the ModelInterpretationSHAP object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for SHAP results
        self.shap_results_dir = os.path.join(self.results_dir, "shap_results")
        os.makedirs(self.shap_results_dir, exist_ok=True)

    def plot_shap_values(self, shap_values, model_name, feature_names):
        """
        Plots and saves SHAP summary plot for a given model.

        :param shap_values: SHAP values for the model's predictions.
        :param model_name: The name of the model (Random Forest or XGBoost).
        :param feature_names: List of feature names.
        """
        # Create a summary plot of SHAP values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, feature_names=feature_names)
        
        # Save the SHAP plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shap_plot_file = os.path.join(self.shap_results_dir, f"shap_summary_{model_name}_{timestamp}.png")
        plt.savefig(shap_plot_file)
        print(f"SHAP summary plot saved as: {shap_plot_file}")

    def interpret_model_predictions(self):
        """
        Interprets the model's predictions using SHAP values for Random Forest and XGBoost.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])  # Features
        y = self.data['TotalClaims']  # Target variable (or use 'TotalPremium' if applicable)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to interpret
        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor()
        }

        # Evaluate each model and interpret using SHAP
        for model_name, model in models.items():
            print(f"Training {model_name} model and interpreting predictions using SHAP...")
            model.fit(X_train, y_train)
            
            # Initialize SHAP explainer
            if model_name == "Random Forest":
                explainer = shap.TreeExplainer(model)  # TreeExplainer is used for tree-based models like RandomForest and XGBoost
            else:
                explainer = shap.Explainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_test)
            
            # Plot and save SHAP values
            self.plot_shap_values(shap_values, model_name, X.columns)

# Example usage of the ModelInterpretationSHAP class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the ModelInterpretationSHAP object
    model_interpretation_shap = ModelInterpretationSHAP(data_path=cleaned_csv_file, results_dir=results_dir)

    # Interpret model predictions using SHAP
    model_interpretation_shap.interpret_model_predictions()