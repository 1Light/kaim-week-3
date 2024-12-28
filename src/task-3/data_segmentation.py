import os
import pandas as pd
from scipy.stats import ttest_ind, chi2_contingency
from datetime import datetime


class DataSegmentation:
    def __init__(self, data_path, results_dir):
        """
        Initializes the DataSegmentation object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for segmentation results
        self.segmentation_dir = os.path.join(self.results_dir, "data_segmentation")
        os.makedirs(self.segmentation_dir, exist_ok=True)

    def log_segmentation_details(self, details, feature_name):
        """
        Logs details about the segmentation process into a file.

        :param details: Details to log.
        :param feature_name: The name of the feature being segmented.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.segmentation_dir, f"{feature_name}_segmentation_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(details)
        print(f"Segmentation details saved as: {log_file}")

    def test_statistical_equivalence(self, group_a, group_b, columns_to_check):
        """
        Tests for statistical equivalence between two groups on given columns.

        :param group_a: Data for Group A (Control Group).
        :param group_b: Data for Group B (Test Group).
        :param columns_to_check: List of columns to test for statistical equivalence.
        :return: Boolean indicating if the groups are statistically equivalent and a details string.
        """
        details = "Statistical Equivalence Testing Results:\n"
        equivalent = True

        for col in columns_to_check:
            if col not in self.data.columns:
                details += f" - Column '{col}' not found in the dataset.\n"
                continue

            if pd.api.types.is_numeric_dtype(self.data[col]):
                # Perform t-test for numerical columns
                t_stat, p_value = ttest_ind(group_a[col].dropna(), group_b[col].dropna(), equal_var=False)
                details += f" - {col}: t-statistic={t_stat:.4f}, p-value={p_value:.4f}\n"
                if p_value < 0.05:
                    equivalent = False
                    details += "   --> Groups are NOT statistically equivalent for this feature.\n"
                else:
                    details += "   --> Groups are statistically equivalent for this feature.\n"
            else:
                # Perform chi-squared test for categorical columns
                contingency_table = pd.crosstab(group_a[col], group_b[col])
                chi2_stat, p_value, _, _ = chi2_contingency(contingency_table)
                details += f" - {col}: chi2-statistic={chi2_stat:.4f}, p-value={p_value:.4f}\n"
                if p_value < 0.05:
                    equivalent = False
                    details += "   --> Groups are NOT statistically equivalent for this feature.\n"
                else:
                    details += "   --> Groups are statistically equivalent for this feature.\n"

        return equivalent, details

    def segment_data(self, feature_column, control_value, test_value, columns_to_check):
        """
        Segments the data into Group A and Group B and ensures statistical equivalence.

        :param feature_column: The feature column for segmentation.
        :param control_value: The value for the control group.
        :param test_value: The value for the test group.
        :param columns_to_check: List of columns to ensure statistical equivalence.
        """
        if feature_column not in self.data.columns:
            raise ValueError(f"Feature column '{feature_column}' not found in the dataset.")

        # Segment data into control and test groups
        group_a = self.data[self.data[feature_column] == control_value]
        group_b = self.data[self.data[feature_column] == test_value]

        details = f"Data Segmentation Results:\n"
        details += f" - Feature: {feature_column}\n"
        details += f" - Control Group Value: {control_value}\n"
        details += f" - Test Group Value: {test_value}\n"
        details += f" - Group A Size: {len(group_a)}\n"
        details += f" - Group B Size: {len(group_b)}\n\n"

        # Test statistical equivalence
        equivalent, equivalence_details = self.test_statistical_equivalence(group_a, group_b, columns_to_check)
        details += equivalence_details

        if equivalent:
            details += "\n--> Groups are statistically equivalent. Proceed with testing.\n"
        else:
            details += "\n--> Groups are NOT statistically equivalent. Adjust your segmentation criteria.\n"

        # Save the segmentation details
        self.log_segmentation_details(details, feature_column)

    def segment_and_analyze(self, feature_column, control_value, test_value, columns_to_check):
        """
        Segments and tests multiple features for statistical equivalence.

        :param feature_column: The feature column to test.
        :param control_value: The value for Group A.
        :param test_value: The value for Group B.
        :param columns_to_check: List of columns to check for statistical equivalence.
        """
        print(f"Segmenting data for feature: {feature_column}")
        self.segment_data(feature_column, control_value, test_value, columns_to_check)


# Example usage of the DataSegmentation class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the DataSegmentation object
    data_segmentation = DataSegmentation(data_path=cleaned_csv_file, results_dir=results_dir)

    # Segment data on a specific feature
    feature_column = "CoverCategory"  # Example segmentation feature
    control_value = "Basic"  # Control group value
    test_value = "Premium"  # Test group value
    columns_to_check = [
        "Age", "Gender", "Citizenship", "TotalPremium", "SumInsured", "VehicleType"
    ]  # Example columns to test for equivalence

    # Perform segmentation and analysis
    data_segmentation.segment_and_analyze(feature_column, control_value, test_value, columns_to_check)