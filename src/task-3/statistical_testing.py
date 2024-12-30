import os
import pandas as pd
import scipy.stats as stats
from datetime import datetime

class StatisticalTesting:
    def __init__(self, data_file, feature_column, target_column):
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        self.data = self.load_data()
        
        self.results_dir = os.path.join(self.base_dir, "../../results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.statistical_test_dir = os.path.join(self.results_dir, "statistical_testing")
        os.makedirs(self.statistical_test_dir, exist_ok=True)

        # Define the feature and target columns
        self.feature_column = feature_column
        self.target_column = target_column

    def load_data(self):
        print(f"Loading data from {self.cleaned_csv_file}...")
        try:
            data = pd.read_csv(self.cleaned_csv_file, low_memory=False)
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.cleaned_csv_file}")
            exit()

    def save_results(self, results, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.statistical_test_dir, f"{filename}_{timestamp}.txt")
        with open(output_file, "w") as file:
            file.write(results)
        print(f"Results saved as: {output_file}")

    def run_statistical_tests(self):
        # Check if the target column exists
        if self.target_column not in self.data.columns:
            print(f"Error: Target column '{self.target_column}' not found in the data.")
            return
        
        # Check if the feature column exists
        if self.feature_column not in self.data.columns:
            print(f"Error: Feature column '{self.feature_column}' not found in the data.")
            return
        
        results = f"=== Statistical Testing for Feature: {self.feature_column} and Target: {self.target_column} ===\n"
        
        # Get data for the two groups in the feature column
        unique_values = self.data[self.feature_column].unique()
        
        if len(unique_values) == 2:
            group_a_value, group_b_value = unique_values
            group_a_data = self.data[self.data[self.feature_column] == group_a_value]
            group_b_data = self.data[self.data[self.feature_column] == group_b_value]

            # Check if the target column is numeric
            if self.data[self.target_column].dtype in ['int64', 'float64']:
                # Perform a t-test for numeric data
                t_stat, p_value = stats.ttest_ind(group_a_data[self.target_column].dropna(),
                                                   group_b_data[self.target_column].dropna(), equal_var=False)
                results += f"T-test for {self.target_column}: T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n"
                
                if p_value < 0.05:
                    results += "Conclusion: Reject the null hypothesis (the feature has a statistically significant impact on the KPI).\n"
                else:
                    results += "Conclusion: Fail to reject the null hypothesis (the feature does not have a significant impact on the KPI).\n"
            
            else:
                # Perform a chi-squared test for categorical data
                contingency_table = pd.crosstab(self.data[self.feature_column], self.data[self.target_column])
                chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
                results += f"Chi-squared test for {self.target_column}: Chi-squared statistic: {chi2_stat:.4f}, P-value: {p_value:.4e}\n"
                
                if p_value < 0.05:
                    results += "Conclusion: Reject the null hypothesis (the feature has a statistically significant impact on the KPI).\n"
                else:
                    results += "Conclusion: Fail to reject the null hypothesis (the feature does not have a significant impact on the KPI).\n"
        else:
            results += f"Error: Feature column '{self.feature_column}' does not have exactly two unique values.\n"

        # Save results
        self.save_results(results, "statistical_testing_results")

if __name__ == "__main__":
    # Example usage:
    # Test the feature 'CoverCategory' with the target column 'TotalPremium'
    statistical_testing = StatisticalTesting(
        data_file="cleaned_ml.csv",  
        feature_column="CoverCategory",  
        target_column="TotalPremium" 
    )
    statistical_testing.run_statistical_tests()