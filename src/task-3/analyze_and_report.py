import os
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from datetime import datetime

class AnalyzeAndReport:
    def __init__(self, data_path, report_dir):
        """
        Initializes the AnalyzeAndReport object.

        :param data_path: Path to the cleaned data CSV file.
        :param report_dir: Path to the directory where reports will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.report_dir = report_dir

        # Ensure the report directory exists
        os.makedirs(self.report_dir, exist_ok=True)

    def analyze_statistical_outcomes(self, group_a, group_b, columns_to_check):
        """
        Analyzes the statistical outcomes and interprets results.

        :param group_a: Data for Group A (Control Group).
        :param group_b: Data for Group B (Test Group).
        :param columns_to_check: List of columns to analyze.
        :return: Analysis details as a string.
        """
        report = "Statistical Analysis Report:\n"
        null_hypotheses_rejected = False

        for col in columns_to_check:
            if col not in self.data.columns:
                report += f" - Column '{col}' not found in the dataset.\n"
                continue

            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Perform t-test for numerical columns
                t_stat, p_value = ttest_ind(group_a[col].dropna(), group_b[col].dropna(), equal_var=False)
                report += f" - {col}: t-statistic={t_stat:.4f}, p-value={p_value:.4f}\n"
                if p_value < 0.05:
                    null_hypotheses_rejected = True
                    report += "   --> Null hypothesis rejected for this feature.\n"
                else:
                    report += "   --> Fail to reject the null hypothesis for this feature.\n"
            else:
                # Perform chi-squared test for categorical columns
                contingency_table = pd.crosstab(group_a[col], group_b[col])
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                report += f" - {col}: chi2-statistic={chi2_stat:.4f}, p-value={p_value:.4f}\n"
                if p_value < 0.05:
                    null_hypotheses_rejected = True
                    report += "   --> Null hypothesis rejected for this feature.\n"
                else:
                    report += "   --> Fail to reject the null hypothesis for this feature.\n"

        report += "\nSummary:\n"
        if null_hypotheses_rejected:
            report += "Some null hypotheses were rejected. This suggests significant differences that may impact business strategy.\n"
        else:
            report += "Failed to reject null hypotheses for all features analyzed. No significant differences detected.\n"

        return report

    def analyze_and_save_report(self, feature_column, control_value, test_value, columns_to_check):
        """
        Analyzes data and saves the report.

        :param feature_column: Feature column for segmentation.
        :param control_value: Control group value.
        :param test_value: Test group value.
        :param columns_to_check: List of columns to analyze.
        """
        if feature_column not in self.data.columns:
            raise ValueError(f"Feature column '{feature_column}' not found in the dataset.")

        # Segment data into control and test groups
        group_a = self.data[self.data[feature_column] == control_value]
        group_b = self.data[self.data[feature_column] == test_value]

        # Analyze outcomes
        report = self.analyze_statistical_outcomes(group_a, group_b, columns_to_check)

        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.report_dir, f"analysis_report_{feature_column}_{timestamp}.txt")
        with open(report_file, "w") as file:
            file.write(report)
        print(f"Analysis report saved at: {report_file}")

# Example usage
if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    report_dir = os.path.join(base_dir, "../../../reports")

    analysis = AnalyzeAndReport(data_path=cleaned_csv_file, report_dir=report_dir)

    feature_column = "CoverCategory"
    control_value = "Basic"
    test_value = "Premium"
    columns_to_check = ["Age", "Gender", "Citizenship", "TotalPremium", "SumInsured", "VehicleType"]

    analysis.analyze_and_save_report(feature_column, control_value, test_value, columns_to_check)