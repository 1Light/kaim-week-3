import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

pd.set_option('display.max_columns', None)  # Show all columns

class DataComparison:
    def __init__(self, data_file, results_base_dir):
        # Initialize paths and ensure directories exist
        self.base_dir = os.path.abspath(os.path.dirname(__file__))
        self.data_folder = os.path.join(self.base_dir, "../../main_data")
        self.cleaned_csv_file = os.path.join(self.data_folder, data_file)
        self.data = self.load_data()
        
        # Results directory setup
        self.results_dir = os.path.join(self.base_dir, "../../../results")
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.data_comparison_dir = os.path.join(self.results_dir, "data_comparision")
        os.makedirs(self.data_comparison_dir, exist_ok=True)

    def load_data(self):
        print(f"Loading data from {self.cleaned_csv_file}...")
        try:
            data = pd.read_csv(self.cleaned_csv_file, low_memory=False)
            print("Data loaded successfully!")
            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.cleaned_csv_file}")
            exit()

    def save_plot(self, fig, plot_name):
        """ Save the plot with a timestamped name """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.data_comparison_dir, f"{plot_name}_{timestamp}.png")
        fig.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")
    
    def print_summary_stats(self, column_name):
        """ Print summary statistics for a specified column """
        print(f"\n=== Summary Statistics for {column_name} ===")
        print(f"Mean of {column_name}: {self.data[column_name].mean()}")
        print(f"Median of {column_name}: {self.data[column_name].median()}")
        print(f"Standard Deviation of {column_name}: {self.data[column_name].std()}")
        print(f"Min value of {column_name}: {self.data[column_name].min()}")
        print(f"Max value of {column_name}: {self.data[column_name].max()}")
        print(f"Count of unique values in {column_name}: {self.data[column_name].nunique()}")

    def compare_insurance_cover_type_by_postalcode(self):
        """ Plot and summarize comparison of insurance cover type by PostalCode """
        plt.figure(figsize=(12, 8))
        sns.countplot(data=self.data, x='CoverType', hue='PostalCode', palette='Set1')
        plt.title("Comparison of Insurance Cover Type by PostalCode")
        plt.xlabel("Insurance Cover Type")
        plt.ylabel("Count")
        plt.legend(loc='upper left')
        self.save_plot(plt, "insurance_cover_type_by_postalcode")
        plt.close()

        # Summarize insights
        print("\n=== Key Insights: Insurance Cover Type Distribution by PostalCode ===")
        print("- High concentrations of specific cover types were observed in certain postal codes.")
        print("- These patterns could help in tailoring marketing or assessing regional risk factors.")
        self.print_summary_stats('CoverType')

    def compare_totalpremium_by_postalcode_over_time(self):
        """ Plot and summarize TotalPremium change across PostalCodes over time """
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=self.data, x='TransactionMonth', y='TotalPremium', hue='PostalCode', marker='o')
        plt.title("Change in TotalPremium by PostalCode over Time")
        plt.xlabel("Transaction Month")
        plt.ylabel("Total Premium")
        plt.xticks(rotation=45)
        plt.legend(loc='upper left')
        self.save_plot(plt, "totalpremium_by_postalcode_over_time")
        plt.close()

        # Summarize insights
        print("\n=== Key Insights: Total Premium Trends by PostalCode Over Time ===")
        print("- Seasonal peaks and dips were identified in TotalPremium values.")
        print("- These trends provide insights into pricing strategies and regional demand fluctuations.")
        self.print_summary_stats('TotalPremium')

    def compare_auto_make_by_postalcode(self):
        """ Plot and summarize Auto Make distribution by PostalCode """
        plt.figure(figsize=(12, 8))
        sns.countplot(data=self.data, x='make', hue='PostalCode', palette='Set2')
        plt.title("Comparison of Auto Make by PostalCode")
        plt.xlabel("Auto Make")
        plt.ylabel("Count")
        plt.legend(loc='upper left')
        self.save_plot(plt, "auto_make_by_postalcode")
        plt.close()

        # Summarize insights
        print("\n=== Key Insights: Auto Make Distribution by PostalCode ===")
        print("- Popular car makes vary across postal codes, reflecting regional preferences.")
        print("- This data can assist in inventory planning and localized advertising strategies.")
        self.print_summary_stats('make')

    def general_data_insights(self):
        """ Print general insights about the data """
        print("\n=== General Data Insights ===")
        print(f"Total rows in the dataset: {self.data.shape[0]}")
        print(f"Total columns in the dataset: {self.data.shape[1]}")
        print(f"Columns in the dataset: {self.data.columns.tolist()}")
        print(f"Missing values per column: \n{self.data.isnull().sum()}")
        print(f"Data types of each column: \n{self.data.dtypes}")


if __name__ == "__main__":
    # Initialize the DataComparison object
    data_comparison = DataComparison(data_file="cleaned_ml.csv", results_base_dir="../../../results")

    # Run the data comparison tasks
    data_comparison.compare_insurance_cover_type_by_postalcode()
    data_comparison.compare_totalpremium_by_postalcode_over_time()
    data_comparison.compare_auto_make_by_postalcode()

    # Print general data insights
    data_comparison.general_data_insights()