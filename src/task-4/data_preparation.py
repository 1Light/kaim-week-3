import os
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

class DataPreparation:
    def __init__(self, data_path, results_dir):
        """
        Initializes the DataPreparation object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for data preparation results
        self.preparation_dir = os.path.join(self.results_dir, "data_preparation")
        os.makedirs(self.preparation_dir, exist_ok=True)

    def log_preparation_details(self, details):
        """
        Logs details about the data preparation process into a file.

        :param details: Details to log.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.preparation_dir, f"preparation_details_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(details)
        print(f"Data preparation details saved as: {log_file}")

    def handle_missing_data(self):
        """
        Handle missing data by imputing or removing missing values based on their nature and the quantity missing.
        """
        # Identify columns with missing values
        missing_data = self.data.isnull().sum()
        missing_columns = missing_data[missing_data > 0].index.tolist()
        details = f"Missing Data Summary:\n{missing_data[missing_columns]}\n\n"

        # Impute missing values for numerical columns with median
        numerical_columns = self.data.select_dtypes(include=["number"]).columns
        imputer = SimpleImputer(strategy="median")
        self.data[numerical_columns] = imputer.fit_transform(self.data[numerical_columns])

        # Impute missing values for categorical columns with the most frequent value
        categorical_columns = self.data.select_dtypes(include=["object"]).columns
        imputer = SimpleImputer(strategy="most_frequent")
        self.data[categorical_columns] = imputer.fit_transform(self.data[categorical_columns])

        details += "Missing data has been imputed.\n"
        self.log_preparation_details(details)

    def feature_engineering(self):
        """
        Create new features that might be relevant to TotalPremium and TotalClaims.
        """
        # Example: Create a new feature based on TotalPremium and SumInsured
        self.data['PremiumToSumRatio'] = self.data['TotalPremium'] / self.data['SumInsured']
        details = f"New feature 'PremiumToSumRatio' created.\n"
        self.log_preparation_details(details)

    def encode_categorical_data(self):
        """
        Encode categorical data into a numeric format using one-hot encoding.
        """
        # Select categorical columns
        categorical_columns = self.data.select_dtypes(include=["object"]).columns

        # One-Hot Encoding for categorical columns
        encoder = OneHotEncoder(drop='first', sparse=False)
        encoded_columns = encoder.fit_transform(self.data[categorical_columns])
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(categorical_columns))

        # Replace the original categorical columns with the encoded ones
        self.data = self.data.drop(columns=categorical_columns)
        self.data = pd.concat([self.data, encoded_df], axis=1)

        details = f"Categorical columns encoded using one-hot encoding.\n"
        self.log_preparation_details(details)

    def train_test_split_data(self):
        """
        Split the data into training and testing sets.
        """
        # Define target variable and features
        X = self.data.drop(columns=['TotalPremium', 'TotalClaims'])
        y = self.data[['TotalPremium', 'TotalClaims']]

        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        details = f"Data split into training and testing sets (80% training, 20% testing).\n"
        self.log_preparation_details(details)
        
        return X_train, X_test, y_train, y_test

    def process_data(self):
        """
        Execute the data preparation steps in order.
        """
        self.handle_missing_data()
        self.feature_engineering()
        self.encode_categorical_data()
        X_train, X_test, y_train, y_test = self.train_test_split_data()
        return X_train, X_test, y_train, y_test


# Example usage of the DataPreparation class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the DataPreparation object
    data_preparation = DataPreparation(data_path=cleaned_csv_file, results_dir=results_dir)

    # Process the data
    X_train, X_test, y_train, y_test = data_preparation.process_data()