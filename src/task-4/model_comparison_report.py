import os
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

class ModelComparisonReport:
    def __init__(self, data_path, results_dir):
        """
        Initializes the ModelComparisonReport object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for model comparison results
        self.comparison_results_dir = os.path.join(self.results_dir, "model_comparison_results")
        os.makedirs(self.comparison_results_dir, exist_ok=True)

    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Evaluates a given model using multiple metrics and generates a performance report.

        :param model: Trained model to evaluate.
        :param X_test: Test features.
        :param y_test: Test labels.
        :param model_name: Name of the model (Random Forest, XGBoost, Linear Regression).
        :return: A dictionary containing evaluation metrics.
        """
        y_pred = model.predict(X_test)

        # For regression models, we use MAE, MSE, RMSE
        if model_name in ["Random Forest", "XGBoost", "Linear Regression"]:
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = mse ** 0.5
            return {
                "Model": model_name,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse
            }

    def compare_models(self):
        """
        Compares the performance of Random Forest, XGBoost, and Linear Regression models.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])  # Features
        y = self.data['TotalClaims']  # Target variable (or use 'TotalPremium' if applicable)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize models
        models = {
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor(),
            "Linear Regression": LinearRegression()
        }

        # Initialize a list to store model performance results
        performance_results = []

        # Evaluate each model
        for model_name, model in models.items():
            print(f"Training and evaluating {model_name} model...")
            model.fit(X_train, y_train)
            model_performance = self.evaluate_model(model, X_test, y_test, model_name)
            performance_results.append(model_performance)

        # Convert performance results to a DataFrame for easy comparison
        performance_df = pd.DataFrame(performance_results)

        # Save performance comparison as a CSV
        comparison_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        performance_report_file = os.path.join(self.comparison_results_dir, f"model_comparison_report_{comparison_timestamp}.csv")
        performance_df.to_csv(performance_report_file, index=False)
        print(f"Model comparison report saved as: {performance_report_file}")

        # Plot the performance metrics
        self.plot_performance_comparison(performance_df)

    def plot_performance_comparison(self, performance_df):
        """
        Plots the comparison of model performance metrics (MAE, MSE, RMSE).

        :param performance_df: DataFrame containing model performance metrics.
        """
        # Plot MAE, MSE, RMSE for each model
        performance_df.set_index('Model', inplace=True)
        performance_df.plot(kind='bar', figsize=(10, 6))

        # Set plot labels and title
        plt.title('Model Comparison: MAE, MSE, RMSE', fontsize=14)
        plt.ylabel('Error Metrics', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the plot
        comparison_plot_file = os.path.join(self.comparison_results_dir, "model_comparison_plot.png")
        plt.savefig(comparison_plot_file)
        print(f"Model comparison plot saved as: {comparison_plot_file}")

# Example usage of the ModelComparisonReport class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the ModelComparisonReport object
    model_comparison_report = ModelComparisonReport(data_path=cleaned_csv_file, results_dir=results_dir)

    # Compare model performances and generate a report
    model_comparison_report.compare_models()