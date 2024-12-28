import os
import pandas as pd
from datetime import datetime


class SelectMetrics:
    def __init__(self, data_path, results_dir):
        """
        Initializes the SelectMetrics object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for KPI analysis results
        self.kpi_dir = os.path.join(self.results_dir, "kpi_analysis")
        os.makedirs(self.kpi_dir, exist_ok=True)

    def log_kpi_details(self, details, kpi_name):
        """
        Logs details about the selected KPI into a file.

        :param details: Details to log.
        :param kpi_name: The name of the KPI being logged.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.kpi_dir, f"{kpi_name}_details_{timestamp}.txt")
        with open(log_file, "w") as file:
            file.write(details)
        print(f"KPI details saved as: {log_file}")

    def select_kpi(self, kpi_column):
        """
        Analyze the selected KPI to measure its impact.

        :param kpi_column: The name of the KPI column in the dataset.
        """
        if kpi_column not in self.data.columns:
            raise ValueError(f"Column '{kpi_column}' not found in the dataset.")

        kpi_data = self.data[kpi_column]
        details = f"Key Performance Indicator: {kpi_column}\n"
        details += f" - Total Observations: {len(kpi_data)}\n"
        details += f" - Missing Values: {kpi_data.isnull().sum()}\n"
        details += f" - Unique Values: {kpi_data.nunique()}\n"

        # Calculate basic statistics for numerical KPIs
        if pd.api.types.is_numeric_dtype(kpi_data):
            details += f" - Mean: {kpi_data.mean():.2f}\n"
            details += f" - Median: {kpi_data.median():.2f}\n"
            details += f" - Standard Deviation: {kpi_data.std():.2f}\n"
            details += f" - Minimum: {kpi_data.min()}\n"
            details += f" - Maximum: {kpi_data.max()}\n"
        else:
            # Analyze distribution for categorical KPIs
            value_counts = kpi_data.value_counts()
            details += f"\nValue Counts:\n{value_counts.to_string()}\n"

        # Save the KPI details
        self.log_kpi_details(details, kpi_column)

    def select_and_analyze_kpis(self, kpi_columns):
        """
        Select and analyze multiple KPIs.

        :param kpi_columns: A list of KPI column names to analyze.
        """
        for kpi in kpi_columns:
            print(f"Analyzing KPI: {kpi}")
            self.select_kpi(kpi)


# Example usage of the SelectMetrics class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the SelectMetrics object
    select_metrics = SelectMetrics(data_path=cleaned_csv_file, results_dir=results_dir)

    # List of KPI columns to analyze
    kpi_columns = ["ProfitMargin", "CustomerSatisfaction", "Revenue", "ConversionRate"]  # Example KPIs

    # Select and analyze the KPIs
    select_metrics.select_and_analyze_kpis(kpi_columns)