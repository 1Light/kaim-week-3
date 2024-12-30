import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from datetime import datetime

class ABHypothesisTesting:
    def __init__(self, data_file):
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        self.data = self.load_data()
        
        self.results_dir = os.path.join(self.base_dir, "../../results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.ab_testing_dir = os.path.join(self.results_dir, "ab_testing")
        os.makedirs(self.ab_testing_dir, exist_ok=True)

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
        output_file = os.path.join(self.ab_testing_dir, f"{filename}_{timestamp}.txt")
        with open(output_file, "w") as file:
            file.write(results)
        print(f"Results saved as: {output_file}")

    def test_risk_differences_across_provinces(self):
        if 'Province' not in self.data.columns or 'TotalPremium' not in self.data.columns or 'TotalClaims' not in self.data.columns:
            print("Error: Required columns not found in the dataset.")
            return

        # Group by province and perform ANOVA to test for risk differences (using TotalPremium or TotalClaims)
        provinces = self.data['Province'].unique()
        groups = [self.data[self.data['Province'] == province]['TotalPremium'] for province in provinces]
        f_stat, p_value = stats.f_oneway(*groups)

        results = (
            "=== Hypothesis Test: Risk Differences Across Provinces ===\n"
            f"F-statistic: {f_stat:.4f}\n"
            f"P-value: {p_value:.4e}\n"
        )
        results += "Conclusion: Reject null hypothesis.\n" if p_value < 0.05 else "Conclusion: Fail to reject null hypothesis.\n"
        self.save_results(results, "risk_differences_across_provinces")

    def test_risk_differences_between_zip_codes(self):
        if 'PostalCode' not in self.data.columns or 'TotalPremium' not in self.data.columns or 'TotalClaims' not in self.data.columns:
            print("Error: Required columns not found in the dataset.")
            return
        
        # Group by postal code and perform ANOVA to test for risk differences
        zip_codes = self.data['PostalCode'].unique()
        groups = [self.data[self.data['PostalCode'] == zip_code]['TotalPremium'] for zip_code in zip_codes]
        f_stat, p_value = stats.f_oneway(*groups)

        results = (
            "=== Hypothesis Test: Risk Differences Between Zip Codes ===\n"
            f"F-statistic: {f_stat:.4f}\n"
            f"P-value: {p_value:.4e}\n"
        )
        results += "Conclusion: Reject null hypothesis.\n" if p_value < 0.05 else "Conclusion: Fail to reject null hypothesis.\n"
        self.save_results(results, "risk_differences_between_zip_codes")

    def test_margin_differences_between_zip_codes(self):
        if 'PostalCode' not in self.data.columns or 'TotalPremium' not in self.data.columns or 'TotalClaims' not in self.data.columns:
            print("Error: Required columns not found in the dataset.")
            return
        
        # Calculate profit margin and group by postal code
        self.data['ProfitMargin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        zip_codes = self.data['PostalCode'].unique()
        groups = [self.data[self.data['PostalCode'] == zip_code]['ProfitMargin'] for zip_code in zip_codes]
        f_stat, p_value = stats.f_oneway(*groups)

        results = (
            "=== Hypothesis Test: Margin Differences Between Zip Codes ===\n"
            f"F-statistic: {f_stat:.4f}\n"
            f"P-value: {p_value:.4e}\n"
        )
        results += "Conclusion: Reject null hypothesis.\n" if p_value < 0.05 else "Conclusion: Fail to reject null hypothesis.\n"
        self.save_results(results, "margin_differences_between_zip_codes")

    def test_risk_differences_between_genders(self):
        if 'Gender' not in self.data.columns or 'TotalPremium' not in self.data.columns or 'TotalClaims' not in self.data.columns:
            print("Error: Required columns not found in the dataset.")
            return
        
        # Separate by gender and perform independent t-test
        male_risk = self.data[self.data['Gender'] == 'Male']['TotalPremium']
        female_risk = self.data[self.data['Gender'] == 'Female']['TotalPremium']
        t_stat, p_value = stats.ttest_ind(male_risk.dropna(), female_risk.dropna(), equal_var=False)

        results = (
            "=== Hypothesis Test: Risk Differences Between Genders ===\n"
            f"T-statistic: {t_stat:.4f}\n"
            f"P-value: {p_value:.4e}\n"
        )
        results += "Conclusion: Reject null hypothesis.\n" if p_value < 0.05 else "Conclusion: Fail to reject null hypothesis.\n"
        self.save_results(results, "risk_differences_between_genders")

    def run_tests(self):
        self.test_risk_differences_across_provinces()
        self.test_risk_differences_between_zip_codes()
        self.test_margin_differences_between_zip_codes()
        self.test_risk_differences_between_genders()


if __name__ == "__main__":
    ab_testing = ABHypothesisTesting(data_file="cleaned_ml.csv")
    ab_testing.run_tests()