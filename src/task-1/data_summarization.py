import os
import pandas as pd

class DataSummarizer:
    def __init__(self, data_file, results_base_dir):
        # Initialize the paths and ensure the results directory exists
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.main_data_dir = os.path.join(self.current_dir, "../../main_data")
        self.results_dir = os.path.join(self.current_dir, "../../../results")
        self.data_file = os.path.join(self.main_data_dir, data_file)
        
        # Ensure the results directory and subfolder exist
        os.makedirs(self.results_dir, exist_ok=True)
        self.data_summarization_dir = os.path.join(self.results_dir, "data_summarization")
        os.makedirs(self.data_summarization_dir, exist_ok=True)
        
        # Load the data
        self.data = self.load_data()

    def load_data(self):
        print(f"Loading data from {self.data_file}...")
        try:
            data = pd.read_csv(self.data_file, low_memory=False)
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_file}")
            exit()

    def generate_descriptive_statistics(self):
        # Descriptive Statistics for Numerical Features
        numerical_features = self.data.select_dtypes(include=["number"])
        descriptive_stats = numerical_features.describe()
        print("\nDescriptive Statistics for Numerical Features:")
        print(descriptive_stats)
        return descriptive_stats

    def calculate_variability(self):
        # Variability (Standard Deviation) for Numerical Features
        numerical_features = self.data.select_dtypes(include=["number"])
        variability = numerical_features.std()
        print("\nVariability (Standard Deviation) for Numerical Features:")
        print(variability)
        return variability

    def print_data_structure(self):
        # Data Structure and Dtypes
        data_structure = self.data.dtypes
        print("\nData Structure and Dtypes:")
        print(data_structure)

    def save_summary(self, summary_data, output_file):
        # Save summary to the data_summarization subfolder
        output_file_path = os.path.join(self.data_summarization_dir, output_file)
        summary_data.to_csv(output_file_path)
        print(f"\nSummary saved to {output_file_path}.")

if __name__ == "__main__":
    # Usage example
    data_summarizer = DataSummarizer(data_file="cleaned_ml.csv", results_base_dir="../../../results")
    
    # Generate and print the statistics
    descriptive_stats = data_summarizer.generate_descriptive_statistics()
    variability = data_summarizer.calculate_variability()
    data_summarizer.print_data_structure()

    # Save the descriptive statistics summary
    data_summarizer.save_summary(descriptive_stats, "data_summary.csv")