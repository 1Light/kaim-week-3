import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set visualization style
sns.set(style="whitegrid")

class DataVisualizer:
    def __init__(self, data_path, results_dir):
        """
        Initializes the DataVisualizer object with the dataset and results directory.

        :param data_path: Path to the cleaned data CSV file.
        :param results_dir: Path to the directory where results will be saved.
        """
        self.data = pd.read_csv(data_path, low_memory=False)
        self.results_dir = results_dir

        # Ensure the results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

        # Create a subfolder for visualization results
        self.visualization_dir = os.path.join(self.results_dir, "visualizations")
        os.makedirs(self.visualization_dir, exist_ok=True)

    def save_plot(self, fig, plot_name):
        """
        Saves a plot to the results directory with a timestamp.

        :param fig: The matplotlib figure to save.
        :param plot_name: The name for the plot file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.visualization_dir, f"{plot_name}_{timestamp}.png")
        fig.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")

    def plot_totalpremium_distribution(self):
        """
        Creates a histogram with KDE for the distribution of TotalPremium.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['TotalPremium'], kde=True, color='blue', bins=30)
        plt.title("Distribution of Total Premium", fontsize=14)
        plt.xlabel("Total Premium", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        self.save_plot(plt, "totalpremium_distribution")
        plt.close()

    def plot_correlation_heatmap(self):
        """
        Creates a correlation heatmap for numeric columns in the dataset.
        """
        numeric_cols = self.data.select_dtypes(include='number').columns
        correlation_matrix = self.data[numeric_cols].corr()

        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Heatmap of Numeric Variables", fontsize=14)
        self.save_plot(plt, "correlation_heatmap")
        plt.close()

    def plot_pairwise_relationships(self):
        """
        Creates a pairplot for selected numeric features, colored by CoverType.
        """
        selected_features = ['TotalPremium', 'Age', 'VehicleAge', 'AnnualMileage']
        sns.pairplot(self.data[selected_features], hue="CoverType", palette="Set2", markers=["o", "s", "D"])
        plt.suptitle("Pairwise Relationships of Key Features by CoverType", fontsize=14)
        self.save_plot(plt, "pairwise_relationships_by_cover_type")
        plt.close()

    def display_eda_insights(self):
        """
        Displays general insights from the Exploratory Data Analysis (EDA).
        """
        print("\n=== Key Insights from EDA ===")
        print(f"Total rows in the dataset: {self.data.shape[0]}")
        print(f"Total columns in the dataset: {self.data.shape[1]}")
        print(f"Columns in the dataset: {self.data.columns.tolist()}")
        print(f"Missing values per column: \n{self.data.isnull().sum()}")
        print(f"Data types of each column: \n{self.data.dtypes}")


# Example usage of the DataVisualizer class
if __name__ == "__main__":
    # Set paths for the data and results
    base_dir = os.path.abspath(os.path.dirname(__file__))  
    data_folder = os.path.join(base_dir, "../../main_data")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
    results_dir = os.path.join(base_dir, "../../../results")

    # Initialize the DataVisualizer object
    visualizer = DataVisualizer(data_path=cleaned_csv_file, results_dir=results_dir)

    # Generate and save visualizations
    visualizer.plot_totalpremium_distribution()
    visualizer.plot_correlation_heatmap()
    visualizer.plot_pairwise_relationships()

    # Display insights from EDA
    visualizer.display_eda_insights()