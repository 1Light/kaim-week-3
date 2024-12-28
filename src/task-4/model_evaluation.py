import os
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class ModelEvaluation:
    def __init__(self, data_path, results_dir):
        """
        Initializes the ModelEvaluation object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for model results
        self.modeling_dir = os.path.join(self.results_dir, "model_evaluation_results")
        os.makedirs(self.modeling_dir, exist_ok=True)

    def log_model_metrics(self, model_name, metrics):
        """
        Logs the evaluation metrics about the model.

        :param model_name: The name of the model.
        :param metrics: The metrics to log.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.modeling_dir, f"{model_name}_metrics_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(metrics)
        print(f"Model metrics saved as: {log_file}")

    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Evaluates the model's performance using accuracy, precision, recall, and F1-score.

        :param model: The machine learning model to evaluate.
        :param X_train: The training features.
        :param X_test: The testing features.
        :param y_train: The training target variable.
        :param y_test: The testing target variable.
        :param model_name: The name of the model.
        """
        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

        # Log the metrics
        metrics = f"Model: {model_name}\n"
        metrics += f"Accuracy: {accuracy:.4f}\n"
        metrics += f"Precision: {precision:.4f}\n"
        metrics += f"Recall: {recall:.4f}\n"
        metrics += f"F1-Score: {f1:.4f}\n"
        
        self.log_model_metrics(model_name, metrics)

    def prepare_and_evaluate(self):
        """
        Prepares the data and evaluates the models: Linear Regression, Random Forests, and XGBoost.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])
        y = self.data[['TotalPremium', 'TotalClaims']]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor()
        }

        # Evaluate each model
        for model_name, model in models.items():
            print(f"Evaluating {model_name} model...")
            self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)


# Example usage of the ModelEvaluation class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the ModelEvaluation object
    model_evaluation = ModelEvaluation(data_path=cleaned_csv_file, results_dir=results_dir)

    # Prepare data and evaluate the models
    model_evaluation.prepare_and_evaluate()