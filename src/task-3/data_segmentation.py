import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

class DataSegmentation:
    def __init__(self, data_file, feature_column, group_a_value, group_b_value):
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        self.data = self.load_data()
        
        self.results_dir = os.path.join(self.base_dir, "../../results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.segmentation_dir = os.path.join(self.results_dir, "data_segmentation")
        os.makedirs(self.segmentation_dir, exist_ok=True)

        # Define the feature and group values
        self.feature_column = feature_column
        self.group_a_value = group_a_value
        self.group_b_value = group_b_value

    def load_data(self):
        print(f"Loading data from {self.cleaned_csv_file}...")
        try:
            data = pd.read_csv(self.cleaned_csv_file, low_memory=False)
            print("Data loaded successfully!")
            print(f"Data preview:\n{data.head()}")  # Preview first few rows
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.cleaned_csv_file}")
            exit()

    def save_results(self, results, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.segmentation_dir, f"{filename}_{timestamp}.txt")
        with open(output_file, "w") as file:
            file.write(results)
        print(f"Results saved as: {output_file}")

    def check_statistical_equivalence(self):
        # Check if there are statistical differences between groups for relevant columns
        relevant_columns = ['Gender', 'Citizenship', 'Province', 'VehicleType', 'SumInsured', 'CoverType']  # Example columns to check
        
        results = "=== Checking Statistical Equivalence Between Groups ===\n"
        
        # Group the data by feature column (Group A vs Group B)
        group_a_data = self.data[self.data[self.feature_column] == self.group_a_value]
        group_b_data = self.data[self.data[self.feature_column] == self.group_b_value]
        print(f"Group A Size: {len(group_a_data)}")
        print(f"Group B Size: {len(group_b_data)}")

        # For numeric columns, use a t-test
        for column in relevant_columns:
            if column in self.data.columns:
                if self.data[column].dtype in ['int64', 'float64']:
                    t_stat, p_value = stats.ttest_ind(group_a_data[column].dropna(), group_b_data[column].dropna(), equal_var=False)
                    results += f"{column} - T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n"
                else:
                    # For categorical columns, use chi-square test
                    contingency_table = pd.crosstab(self.data[self.feature_column], self.data[column])
                    chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                    results += f"{column} - Chi-square statistic: {chi2_stat:.4f}, P-value: {p_value:.4e}\n"
        
        # Save results
        self.save_results(results, "statistical_equivalence_check")

    def check_feature_values(self):
        # Check the unique values in the feature column to ensure correct segmentation
        unique_values = self.data[self.feature_column].unique()
        print(f"Unique values in '{self.feature_column}' column: {unique_values}")
        
    def segment_data(self):
        # Segment the data into Group A and Group B
        group_a = self.data[self.data[self.feature_column] == self.group_a_value]
        group_b = self.data[self.data[self.feature_column] == self.group_b_value]
        
        group_a.to_csv(os.path.join(self.segmentation_dir, "group_a.csv"), index=False)
        group_b.to_csv(os.path.join(self.segmentation_dir, "group_b.csv"), index=False)
        
        print(f"Group A and Group B data saved as group_a.csv and group_b.csv in the segmentation directory.")

    def run_segmentation(self):
        print("Starting data segmentation process...\n")
        self.check_feature_values()  # Check feature column values
        self.check_statistical_equivalence()
        self.segment_data()
        print("\nData segmentation process completed!")

if __name__ == "__main__":
    # Example usage:
    # Test the feature 'CoverCategory', splitting it into 'Comprehensive' (Group A) and 'Third Party' (Group B)
    data_segmentation = DataSegmentation(data_file="cleaned_ml.csv", feature_column="CoverCategory", group_a_value="Third party", group_b_value="Own damage")
    data_segmentation.run_segmentation()