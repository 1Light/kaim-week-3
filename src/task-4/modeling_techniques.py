import os
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class ModelingTechniques:
    def __init__(self, data_path, results_dir):
        """
        Initializes the ModelingTechniques object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for model results
        self.modeling_dir = os.path.join(self.results_dir, "modeling_results")
        os.makedirs(self.modeling_dir, exist_ok=True)

    def log_model_details(self, model_name, details):
        """
        Logs details about the model and its performance into a file.

        :param model_name: The name of the model.
        :param details: Details to log.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.modeling_dir, f"{model_name}_results_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(details)
        print(f"Model details saved as: {log_file}")

    def train_and_evaluate(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Trains a model and evaluates its performance.

        :param model: The machine learning model to train.
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

        # Evaluate performance
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        details = f"Model: {model_name}\n"
        details += f"Mean Squared Error (MSE): {mse:.4f}\n"
        details += f"R-squared: {r2:.4f}\n"
        
        self.log_model_details(model_name, details)

    def prepare_and_model(self):
        """
        Prepare the data and apply different models: Linear Regression, Decision Trees, Random Forests, XGBoost.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])
        y = self.data[['TotalPremium', 'TotalClaims']]

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Models to evaluate
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": xgb.XGBRegressor()
        }

        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"Training {model_name} model...")
            self.train_and_evaluate(model, X_train, X_test, y_train, y_test, model_name)


# Example usage of the ModelingTechniques class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the ModelingTechniques object
    modeling_techniques = ModelingTechniques(data_path=cleaned_csv_file, results_dir=results_dir)

    # Prepare data and apply modeling techniques
    modeling_techniques.prepare_and_model()