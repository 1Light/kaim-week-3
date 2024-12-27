import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set(style="whitegrid")

class UnivariateAnalysis:
    def __init__(self, data_path, results_dir):
        """
        Initializes the UnivariateAnalysis object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for univariate analysis results
        self.univariate_dir = os.path.join(self.results_dir, "univariate")
        os.makedirs(self.univariate_dir, exist_ok=True)

    def save_plot(self, fig, plot_name):
        """
        Saves a plot to the results directory with a timestamp.

        :param fig: The matplotlib figure to save.
        :param plot_name: The name for the plot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.univariate_dir, f"{plot_name}_{timestamp}.png")
        fig.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")

    def print_basic_info(self):
        """
        Prints basic information about the dataset including its shape, data types, and missing values.
        """
        print("Basic Information about the dataset:")
        print(self.data.info())

        missing_values = self.data.isnull().sum()
        print("\nMissing values per column:")
        print(missing_values[missing_values > 0])  # Display columns with missing values

    def analyze_numerical_columns(self):
        """
        Analyzes numerical columns by printing basic statistics and visualizing their distributions.
        """
        numerical_cols = self.data.select_dtypes(include=['number']).columns
        print("\nNumerical Columns Analysis:")

        for col in numerical_cols:
            print(f"\nAnalysis for {col}:")
            print(f" - Missing values: {self.data[col].isnull().sum()}")
            print(f" - Unique values: {self.data[col].nunique()}")
            print(f" - Mean: {self.data[col].mean():.2f}")
            print(f" - Standard Deviation: {self.data[col].std():.2f}")
            print(f" - Min: {self.data[col].min()}")
            print(f" - Max: {self.data[col].max()}")

            # Plot the distribution of the numerical column
            plt.figure(figsize=(10, 6))
            sns.histplot(self.data[col], kde=True, bins=30, color='skyblue')
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Frequency')
            self.save_plot(plt, f"histogram_{col}")
            plt.close()

    def analyze_categorical_columns(self):
        """
        Analyzes categorical columns by printing basic statistics and visualizing their distributions.
        """
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        print("\nCategorical Columns Analysis:")

        for col in categorical_cols:
            print(f"\nAnalysis for {col}:")
            print(f" - Missing values: {self.data[col].isnull().sum()}")
            print(f" - Unique values: {self.data[col].nunique()}")

            # Plot the distribution of the categorical column
            plt.figure(figsize=(10, 6))
            sns.countplot(data=self.data, x=col)  # Removed the palette argument
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            self.save_plot(plt, f"bar_chart_{col}")
            plt.close()

    def perform_univariate_analysis(self):
        """
        Perform the full univariate analysis, including basic info, numerical and categorical column analysis.
        """
        self.print_basic_info()
        self.analyze_numerical_columns()
        self.analyze_categorical_columns()


# Example usage of the UnivariateAnalysis class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the UnivariateAnalysis object
    univariate_analysis = UnivariateAnalysis(data_path=cleaned_csv_file, results_dir=results_dir)

    # Perform univariate analysis
    univariate_analysis.perform_univariate_analysis()