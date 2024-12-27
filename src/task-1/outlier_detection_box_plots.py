import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set(style="whitegrid")

class OutlierDetector:
    def __init__(self, data_path, results_dir):
        """
        Initializes the OutlierDetector object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for outlier detection results
        self.outlier_dir = os.path.join(self.results_dir, "outlier")
        os.makedirs(self.outlier_dir, exist_ok=True)

    def save_plot(self, fig, plot_name):
        """
        Saves a plot to the results directory with a timestamp.

        :param fig: The matplotlib figure to save.
        :param plot_name: The name for the plot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.outlier_dir, f"{plot_name}_{timestamp}.png")
        fig.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")

    def detect_outliers(self):
        """
        Detects and visualizes outliers in the dataset using boxplots.
        Prints outlier statistics for each numerical column.
        """
        numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_columns:
            # Create box plot for outlier detection
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=self.data, x=col)
            plt.title(f"Box Plot of {col} - Outlier Detection")
            plt.xlabel(col)
            self.save_plot(plt, f"outlier_detection_{col}")
            plt.close()

            # Calculate and print outlier statistics
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.data[(self.data[col] < lower_bound) | (self.data[col] > upper_bound)]
            print(f"\n=== Outlier Analysis for {col} ===")
            print(f"- Q1 (25th percentile): {Q1}")
            print(f"- Q3 (75th percentile): {Q3}")
            print(f"- IQR (Interquartile Range): {IQR}")
            print(f"- Lower Bound for Outliers: {lower_bound}")
            print(f"- Upper Bound for Outliers: {upper_bound}")
            print(f"- Number of Outliers Detected: {outliers.shape[0]}")
            print("- Suggestions:")
            print("  * Investigate the context of detected outliers to determine their validity.")
            print("  * Consider handling outliers using transformations, filtering, or robust models if needed.")

# Example usage of the OutlierDetector class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")  # Updated cleaned file name
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the OutlierDetector object
    outlier_detector = OutlierDetector(data_path=cleaned_csv_file, results_dir=results_dir)

    # Perform outlier detection and visualization
    outlier_detector.detect_outliers()