import os
import pandas as pd
from datetime import datetime
import scipy.stats as stats

class StatisticalAnalysisReport:
    def __init__(self, data_file, feature_column, target_column, significance_level=0.05):
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        self.data = self.load_data()
        
        # Define the feature and target columns for testing
        self.feature_column = feature_column
        self.target_column = target_column
        self.significance_level = significance_level
        
        self.results_dir = os.path.join(self.base_dir, "../../results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.report_dir = os.path.join(self.results_dir, "statistical_analysis_reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def load_data(self):
        print(f"Loading data from {self.cleaned_csv_file}...")
        try:
            data = pd.read_csv(self.cleaned_csv_file, low_memory=False)
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.cleaned_csv_file}")
            exit()

    def save_report(self, report, filename):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.report_dir, f"{filename}_{timestamp}.txt")
        with open(output_file, "w") as file:
            file.write(report)
        print(f"Report saved as: {output_file}")
    
    def perform_statistical_test(self):
        # Choose an appropriate test based on the type of data
        results = f"=== Statistical Analysis Report ===\n\n"
        results += f"Feature: {self.feature_column}, Target: {self.target_column}\n"
        results += f"Significance Level: {self.significance_level}\n\n"
        
        # Check the types of columns
        feature_data = self.data[self.feature_column]
        target_data = self.data[self.target_column]

        # Check if the feature is categorical or numerical
        if feature_data.dtype == 'object':  # Categorical feature
            contingency_table = pd.crosstab(feature_data, target_data)
            chi2_stat, p_value, _, _ = stats.chi2_contingency(contingency_table)
            results += f"Chi-square statistic: {chi2_stat:.4f}, P-value: {p_value:.4e}\n"
            if p_value < self.significance_level:
                results += "Reject the null hypothesis: There is a significant relationship between the feature and the target.\n"
            else:
                results += "Fail to reject the null hypothesis: There is no significant relationship between the feature and the target.\n"
        
        elif feature_data.dtype in ['int64', 'float64']:  # Numerical feature
            t_stat, p_value = stats.ttest_ind(feature_data.dropna(), target_data.dropna(), equal_var=False)
            results += f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4e}\n"
            if p_value < self.significance_level:
                results += "Reject the null hypothesis: The feature has a significant impact on the target.\n"
            else:
                results += "Fail to reject the null hypothesis: The feature does not have a significant impact on the target.\n"
        
        return results
    
    def analyze_and_report(self):
        # Perform statistical test
        statistical_results = self.perform_statistical_test()

        # Interpret the results and write the report
        interpretation = f"\n=== Interpretation ===\n"
        
        # Add business impact interpretation based on the p-value
        if "Reject the null hypothesis" in statistical_results:
            interpretation += f"Feature: {self.feature_column} shows a significant impact on {self.target_column}. This suggests that {self.feature_column} should be considered as an important factor in business strategy to optimize {self.target_column}.\n"
            interpretation += f"Customer experience could also benefit from this, by focusing on {self.feature_column} to drive improvements.\n"
        else:
            interpretation += f"Feature: {self.feature_column} does not show a significant impact on {self.target_column}. This suggests that {self.feature_column} may not need to be prioritized in business strategy for improving {self.target_column}.\n"
            interpretation += f"However, further exploration of other features could reveal areas for improvement.\n"
        
        # Combine the statistical results with interpretation
        full_report = statistical_results + interpretation
        
        # Save the report
        self.save_report(full_report, "statistical_analysis_report")
    
if __name__ == "__main__":
    # Example usage:
    # Test the impact of 'Gender' on 'TotalPremium'
    analysis_report = StatisticalAnalysisReport(data_file="cleaned_ml.csv", feature_column="Gender", target_column="TotalPremium")
    analysis_report.analyze_and_report()