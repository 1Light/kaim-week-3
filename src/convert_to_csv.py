import os
import pandas as pd

class DataConverter:
    def __init__(self, base_dir, input_filename="data/ml.txt", output_filename="ml.csv"):
        """
        Initializes the DataConverter object with paths for input and output files.
        
        :param base_dir: The base directory for relative paths.
        :param input_filename: The name of the input file to load.
        :param output_filename: The name of the output CSV file.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.data_folder = os.path.join(self.base_dir, "../")
        self.input_file = os.path.join(self.data_folder, input_filename)
        self.main_data_folder = os.path.join(self.data_folder, "main_data")
        self.output_file = os.path.join(self.main_data_folder, output_filename)

        # Ensure the main_data folder exists
        os.makedirs(self.main_data_folder, exist_ok=True)

    def load_data(self):
        """
        Loads the data from the input file.
        
        :return: The loaded data as a pandas DataFrame.
        """
        try:
            print(f"Loading data from {self.input_file}...")
            # Read the pipe-separated file
            data = pd.read_csv(self.input_file, delimiter="|")
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: The file {self.input_file} does not exist. Please ensure the path is correct.")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
        return None

    def convert_and_save(self, data):
        """
        Converts the loaded data to CSV format and saves it to the specified output file.
        
        :param data: The data to convert and save.
        """
        if data is not None:
            try:
                print(f"Converting to CSV and saving to {self.output_file}...")
                data.to_csv(self.output_file, index=False)
                print(f"File successfully converted and stored in {self.main_data_folder}!")
            except Exception as e:
                print(f"An error occurred while saving the data: {e}")

    def process(self):
        """
        Orchestrates the loading, conversion, and saving of the data.
        """
        data = self.load_data()
        self.convert_and_save(data)

# Example usage of the DataConverter class
if __name__ == "__main__":
    # Initialize the DataConverter object
    converter = DataConverter(base_dir=os.path.dirname(__file__))

    # Perform the data conversion process
    converter.process()