import os
import pandas as pd
import scipy.stats as stats
from datetime import datetime


class HypothesisTesting:
    def __init__(self, data_path, results_dir):
        """
        Initializes the HypothesisTesting object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for hypothesis testing results
        self.hypothesis_dir = os.path.join(self.results_dir, "hypothesis_testing")
        os.makedirs(self.hypothesis_dir, exist_ok=True)

    def log_results(self, results, test_name):
        """
        Saves test results to a log file with a timestamp.

        :param results: The string results to save.
        :param test_name: The name of the hypothesis test.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.hypothesis_dir, f"{test_name}_results_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(results)
        print(f"Results saved as: {log_file}")

    def perform_anova(self, groups, label):
        """
        Perform ANOVA for given groups.

        :param groups: A list of groups for comparison.
        :param label: The label for the test.
        """
        anova_result = stats.f_oneway(*groups)
        result_str = f"ANOVA result for {label}: p-value = {anova_result.pvalue}\n"
        if anova_result.pvalue < 0.05:
            result_str += f"Reject the null hypothesis: Significant differences found in {label}.\n\n"
        else:
            result_str += f"Fail to reject the null hypothesis: No significant differences in {label}.\n\n"

        self.log_results(result_str, f"anova_{label}")

    def perform_ttest(self, group1, group2, label):
        """
        Perform t-test for two groups.

        :param group1: Data for group 1.
        :param group2: Data for group 2.
        :param label: The label for the test.
        """
        t_test_result = stats.ttest_ind(group1, group2, equal_var=False)
        result_str = f"T-test result for {label}: p-value = {t_test_result.pvalue}\n"
        if t_test_result.pvalue < 0.05:
            result_str += f"Reject the null hypothesis: Significant differences found in {label}.\n\n"
        else:
            result_str += f"Fail to reject the null hypothesis: No significant differences in {label}.\n\n"

        self.log_results(result_str, f"ttest_{label}")

    def perform_hypothesis_testing(self):
        """
        Perform hypothesis testing on the dataset.
        """
        # Check for required columns
        required_columns = ["Province", "PostalCode", "Gender", "Risk", "ProfitMargin"]
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"The dataset must include the following columns: {', '.join(required_columns)}")

        # Drop rows with missing values in relevant columns
        self.data = self.data.dropna(subset=required_columns)

        # 1. Test: "There are no risk differences across provinces"
        province_groups = [group["Risk"] for _, group in self.data.groupby("Province")]
        self.perform_anova(province_groups, "Risk across Provinces")

        # 2. Test: "There are no risk differences between zip codes"
        zip_groups = [group["Risk"] for _, group in self.data.groupby("PostalCode")]
        self.perform_anova(zip_groups, "Risk across Postal Codes")

        # 3. Test: "There are no significant margin (profit) differences between zip codes"
        profit_groups = [group["ProfitMargin"] for _, group in self.data.groupby("PostalCode")]
        self.perform_anova(profit_groups, "Profit Margins across Postal Codes")

        # 4. Test: "There are no significant risk differences between Women and Men"
        female_risk = self.data[self.data["Gender"] == "Female"]["Risk"]
        male_risk = self.data[self.data["Gender"] == "Male"]["Risk"]
        self.perform_ttest(female_risk, male_risk, "Risk between Women and Men")


# Example usage of the HypothesisTesting class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the HypothesisTesting object
    hypothesis_testing = HypothesisTesting(data_path=cleaned_csv_file, results_dir=results_dir)

    # Perform hypothesis testing
    hypothesis_testing.perform_hypothesis_testing()