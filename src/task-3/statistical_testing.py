import os
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency, zscore
from datetime import datetime


class StatisticalTesting:
    def __init__(self, data_path, results_dir):
        """
        Initializes the StatisticalTesting object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for statistical testing results
        self.testing_dir = os.path.join(self.results_dir, "statistical_testing")
        os.makedirs(self.testing_dir, exist_ok=True)

    def log_test_results(self, results, test_name):
        """
        Logs the results of statistical tests into a file.

        :param results: Results to log.
        :param test_name: The name of the statistical test performed.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.testing_dir, f"{test_name}_results_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(results)
        print(f"Test results saved as: {log_file}")

    def perform_t_test(self, group_a, group_b, kpi_column):
        """
        Performs a t-test for numerical data between two groups.

        :param group_a: Data for Group A (Control Group).
        :param group_b: Data for Group B (Test Group).
        :param kpi_column: The column representing the KPI.
        :return: A string summarizing the results.
        """
        t_stat, p_value = ttest_ind(group_a[kpi_column].dropna(), group_b[kpi_column].dropna(), equal_var=False)
        result = f"T-Test Results for KPI: {kpi_column}\n"
        result += f" - t-statistic: {t_stat:.4f}\n"
        result += f" - p-value: {p_value:.4f}\n"
        if p_value < 0.05:
            result += " --> p-value < 0.05: Reject the null hypothesis. The feature has a statistically significant effect on the KPI.\n"
        else:
            result += " --> p-value >= 0.05: Fail to reject the null hypothesis. The feature does not have a statistically significant effect on the KPI.\n"
        return result

    def perform_chi_squared_test(self, group_a, group_b, kpi_column):
        """
        Performs a chi-squared test for categorical data between two groups.

        :param group_a: Data for Group A (Control Group).
        :param group_b: Data for Group B (Test Group).
        :param kpi_column: The column representing the KPI.
        :return: A string summarizing the results.
        """
        contingency_table = pd.crosstab(group_a[kpi_column], group_b[kpi_column])
        chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
        result = f"Chi-Squared Test Results for KPI: {kpi_column}\n"
        result += f" - chi2-statistic: {chi2_stat:.4f}\n"
        result += f" - p-value: {p_value:.4f}\n"
        if p_value < 0.05:
            result += " --> p-value < 0.05: Reject the null hypothesis. The feature has a statistically significant effect on the KPI.\n"
        else:
            result += " --> p-value >= 0.05: Fail to reject the null hypothesis. The feature does not have a statistically significant effect on the KPI.\n"
        return result

    def conduct_statistical_tests(self, feature_column, kpi_column, control_value, test_value):
        """
        Conducts statistical tests to evaluate the impact of a feature on the KPI.

        :param feature_column: The feature column for segmentation.
        :param kpi_column: The column representing the KPI.
        :param control_value: The value for the control group.
        :param test_value: The value for the test group.
        """
        if feature_column not in self.data.columns or kpi_column not in self.data.columns:
            raise ValueError(f"Feature column '{feature_column}' or KPI column '{kpi_column}' not found in the dataset.")

        # Segment data into control and test groups
        group_a = self.data[self.data[feature_column] == control_value]
        group_b = self.data[self.data[feature_column] == test_value]

        result_summary = f"Statistical Testing Results:\n"
        result_summary += f" - Feature: {feature_column}\n"
        result_summary += f" - KPI: {kpi_column}\n"
        result_summary += f" - Control Group Value: {control_value}\n"
        result_summary += f" - Test Group Value: {test_value}\n"
        result_summary += f" - Group A Size: {len(group_a)}\n"
        result_summary += f" - Group B Size: {len(group_b)}\n\n"

        # Choose the appropriate test based on KPI data type
        if pd.api.types.is_numeric_dtype(self.data[kpi_column]):
            result_summary += self.perform_t_test(group_a, group_b, kpi_column)
        else:
            result_summary += self.perform_chi_squared_test(group_a, group_b, kpi_column)

        # Log the test results
        self.log_test_results(result_summary, f"{feature_column}_vs_{kpi_column}")

    def test_feature_impact(self, feature_column, kpi_column, control_value, test_value):
        """
        High-level method to test the statistical significance of a feature on the KPI.

        :param feature_column: The feature column to test.
        :param kpi_column: The column representing the KPI.
        :param control_value: The value for the control group.
        :param test_value: The value for the test group.
        """
        print(f"Conducting statistical tests for feature: {feature_column} on KPI: {kpi_column}")
        self.conduct_statistical_tests(feature_column, kpi_column, control_value, test_value)


# Example usage of the StatisticalTesting class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the StatisticalTesting object
    statistical_testing = StatisticalTesting(data_path=cleaned_csv_file, results_dir=results_dir)

    # Specify the feature and KPI for testing
    feature_column = "CoverCategory"  # Example feature
    kpi_column = "TotalPremium"  # Example KPI
    control_value = "Basic"  # Control group value
    test_value = "Premium"  # Test group value

    # Perform statistical testing
    statistical_testing.test_feature_impact(feature_column, kpi_column, control_value, test_value)