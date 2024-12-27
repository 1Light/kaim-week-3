import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class DataAnalyzer:
    def __init__(self, data_file, results_dir):
        self.data_file = data_file
        self.results_dir = os.path.abspath(results_dir)
        self.data = self._load_data()
        self.bivariate_dir = self._create_results_subdir("bivariate")

    def _load_data(self):
        """Load the dataset from the provided file."""
        try:
            return pd.read_csv(self.data_file, low_memory=False)
        except FileNotFoundError:
            print(f"Error: File not found - {self.data_file}")
            raise

    def _create_results_subdir(self, subdir_name):
        """Create a subdirectory for storing analysis results."""
        subdir_path = os.path.join(self.results_dir, subdir_name)
        os.makedirs(subdir_path, exist_ok=True)
        return subdir_path

    def save_plot(self, fig, plot_name):
        """Save the plot with a timestamped filename."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(self.bivariate_dir, f"{plot_name}_{timestamp}.png")
        fig.savefig(plot_filename)
        print(f"Plot saved as: {plot_filename}")

    def scatter_plot(self, x, y, hue, title, xlabel, ylabel, legend_location="upper left"):
        """Generate and save a scatter plot."""
        plt.figure(figsize=(12, 8))
        scatter_plot = sns.scatterplot(data=self.data, x=x, y=y, hue=hue, palette="Set1", alpha=0.7)
        scatter_plot.set_title(title)
        scatter_plot.set_xlabel(xlabel)
        scatter_plot.set_ylabel(ylabel)
        scatter_plot.legend(loc=legend_location, bbox_to_anchor=(1, 1), title=hue)
        self.save_plot(plt, f"scatter_{x}_vs_{y}_by_{hue}")
        plt.close()

    def correlation_matrix(self, columns, title):
        """Generate and save a correlation matrix."""
        corr_matrix = self.data[columns].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(title)
        self.save_plot(plt, f"correlation_matrix_{'_'.join(columns)}")
        plt.close()
        return corr_matrix

    def print_correlation_insights(self, corr_matrix, columns):
        """Print insights from the correlation matrix."""
        print("\n=== Insights from Correlation Matrix ===")
        for col1 in columns:
            for col2 in columns:
                if col1 != col2:
                    correlation = corr_matrix.loc[col1, col2]
                    print(f"  * Correlation between {col1} and {col2}: {correlation:.2f}")
        print("- Strong correlations indicate potential relationships to explore further.")
        print("- Weak correlations may suggest independence or non-linear relationships.")

    def print_scatter_insights(self, x, y):
        """Print insights for the scatter plot."""
        print("\n=== Insights from Scatter Plot ===")
        print(f"- Analyzing the scatter plot of {x} vs. {y} can reveal clusters or outliers.")
        print(f"- Check if higher {x} values correlate with higher {y} values.")
        print(f"- Categories with more concentrated points might have more consistent behavior.")


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.dirname(__file__))
    data_folder = os.path.join(base_dir, "../../main_data")
    results_folder = os.path.join(base_dir, "../../../results")
    cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")

    analyzer = DataAnalyzer(cleaned_csv_file, results_folder)

    # Scatter plot analysis
    analyzer.scatter_plot(
        x="TotalPremium",
        y="TotalClaims",
        hue="PostalCode",
        title="Scatter Plot of TotalPremium vs TotalClaims by PostalCode",
        xlabel="TotalPremium",
        ylabel="TotalClaims",
    )
    analyzer.print_scatter_insights("TotalPremium", "TotalClaims")

    # Correlation matrix analysis
    correlation_columns = ['TotalPremium', 'TotalClaims']
    corr_matrix = analyzer.correlation_matrix(correlation_columns, "Correlation Matrix for TotalPremium and TotalClaims")
    analyzer.print_correlation_insights(corr_matrix, correlation_columns)