import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataPreparation:
    def __init__(self, data_file):
        # Define the base directory and data folder
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")  # Update with your actual folder structure
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        
        # Load the data
        self.data = self.load_data()

    def load_data(self):
        """
        Load dataset into pandas DataFrame with error handling for file not found.
        """
        print(f"Loading data from {self.cleaned_csv_file}...")
        try:
            data = pd.read_csv(self.cleaned_csv_file, low_memory=False)
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.cleaned_csv_file}")
            exit()

    def feature_engineering(self):
        """
        Create new features relevant to TotalPremium and TotalClaims.
        This is a sample; you can add more engineering logic based on business needs.
        """
        print("Performing feature engineering...")
        
        # Example: Create a new feature for 'Premium to Claims Ratio'
        self.data['PremiumToClaimsRatio'] = self.data['TotalPremium'] / (self.data['TotalClaims'] + 1e-5)  # Avoid division by zero

        print("Feature engineering completed.")
    
    def encode_categorical_data(self):
        """
        Convert categorical data into numeric format using OneHotEncoding.
        """
        print("Encoding categorical data...")

        categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        # Define the transformers for encoding categorical columns
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])

        # Apply OneHotEncoding to categorical columns
        self.data_encoded = pd.DataFrame(
            ColumnTransformer(
                transformers=[('cat', categorical_transformer, categorical_columns)]
            ).fit_transform(self.data)
        )
        
        # Update column names (handling one-hot encoding column names)
        encoded_columns = [f"{col}_{category}" for col in categorical_columns for category in self.data[col].unique()][:-1]
        self.data_encoded.columns = self.data.columns.tolist() + encoded_columns

        print("Categorical data encoding completed.")

    def split_data(self, test_size=0.3):
        """
        Split the data into training and test sets.
        """
        print("Splitting data into train and test sets...")

        # Define features and target variables
        X = self.data_encoded.drop(['TotalPremium', 'TotalClaims'], axis=1)  # Drop target variables
        y = self.data[['TotalPremium', 'TotalClaims']]  # Target variables
        
        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        print(f"Data split into {len(X_train)} training samples and {len(X_test)} test samples.")
        return X_train, X_test, y_train, y_test

    def prepare_data(self):
        """
        Prepare data by performing feature engineering, encoding, and splitting.
        """
        self.feature_engineering()
        self.encode_categorical_data()
        return self.split_data()

if __name__ == "__main__":
    # Initialize DataPreparation object with the correct data file path
    data_prep = DataPreparation(data_file='cleaned_ml.csv')
    
    # Prepare the data
    X_train, X_test, y_train, y_test = data_prep.prepare_data()