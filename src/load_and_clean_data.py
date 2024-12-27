import os
import pandas as pd

class DataCleaner:
    def __init__(self, base_dir, input_filename="ml.csv", output_filename="cleaned_ml.csv"):
        """
        Initializes the DataCleaner object with paths for input and output files.
        
        :param base_dir: The base directory for relative paths.
        :param input_filename: The name of the input CSV file to load.
        :param output_filename: The name of the output CSV file after cleaning.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.data_folder = os.path.join(self.base_dir, "../main_data")
        self.input_file = os.path.join(self.data_folder, input_filename)
        self.output_file = os.path.join(self.data_folder, output_filename)
        self.data = None

    def load_data(self):
        """
        Loads the data from the input CSV file into a pandas DataFrame.
        
        :return: The loaded data as a pandas DataFrame.
        """
        try:
            print(f"Loading data from {self.input_file}...")
            self.data = pd.read_csv(self.input_file, low_memory=False)
            print("Data loaded successfully!")
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist. Please ensure the path is correct.")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def assess_data_quality(self):
        """
        Performs a data quality assessment, including checking for missing values and empty columns.
        """
        if self.data is not None:
            print("\nData Quality Assessment: Checking for missing values...")
            missing_data = self.data.isnull().sum()
            missing_percentage = (missing_data / len(self.data)) * 100
            missing_summary = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
            missing_summary = missing_summary[missing_summary['Missing Values'] > 0]

            if not missing_summary.empty:
                print("\nColumns with Missing Data:")
                print(missing_summary)
            else:
                print("\nNo missing values detected.")

            empty_columns = self.data.columns[self.data.isnull().all()]
            print(f"\nEmpty Columns (all values are NaN): {list(empty_columns)}")
            self.data = self.data.drop(columns=empty_columns)

    def clean_data(self):
        """
        Cleans the data by converting data types, filling missing values, and standardizing values.
        """
        if self.data is not None:
            # Convert 'CapitalOutstanding' to numeric
            self.data['CapitalOutstanding'] = pd.to_numeric(self.data['CapitalOutstanding'], errors='coerce')

            # Convert 'TransactionMonth' and 'VehicleIntroDate' to datetime
            self.data['TransactionMonth'] = pd.to_datetime(self.data['TransactionMonth'], errors='coerce')
            
            # Handle 'VehicleIntroDate' specifically
            def parse_vehicle_intro_date(date_value):
                try:
                    return pd.to_datetime(self.data['VehicleIntroDate'], format='%m/%Y', errors='coerce')
                except Exception:
                    return pd.to_datetime(date_value, errors='coerce')
            
            # Convert numeric-like columns to integers
            integer_columns = ['Cylinders', 'NumberOfDoors']
            for col in integer_columns:
                self.data[col] = pd.to_numeric(self.data[col], downcast='integer', errors='coerce')

            # Convert postal codes to strings
            self.data['PostalCode'] = self.data['PostalCode'].astype(str)

            # Convert boolean-like columns to booleans
            boolean_columns = [
                'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding', 'NewVehicle', 
                'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder'
            ]
            for col in boolean_columns:
                self.data[col] = self.data[col].astype(str).str.strip().str.lower().map({'yes': True, 'no': False})

            # Apply consistent formatting to object columns
            for col, dtype in self.data.dtypes.items():
                if dtype == "object":
                    self.data[col] = self.data[col].astype(str).str.strip()
                    self.data[col] = self.data[col].str.capitalize()

            # Fill missing values for categorical and numerical columns
            for col in self.data.columns:
                if self.data[col].dtype == 'object':
                    mode_value = self.data[col].mode()[0]
                    self.data[col] = self.data[col].fillna(mode_value)
                elif self.data[col].dtype in ['float64', 'int64']:
                    median_value = self.data[col].median()
                    self.data[col] = self.data[col].fillna(median_value)

    def save_cleaned_data(self):
        """
        Saves the cleaned data to the output CSV file.
        """
        if self.data is not None:
            self.data.to_csv(self.output_file, index=False)
            print(f"Cleaned data saved to {self.output_file}!")

    def process(self):
        """
        Orchestrates the entire process of loading, assessing, cleaning, and saving the data.
        """
        self.load_data()
        self.assess_data_quality()
        self.clean_data()
        self.save_cleaned_data()

# Example usage of the DataCleaner class
if __name__ == "__main__":
    # Initialize the DataCleaner object
    cleaner = DataCleaner(base_dir=os.path.dirname(__file__))

    # Perform the data processing
    cleaner.process()