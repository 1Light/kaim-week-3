import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the cleaned data
base_dir = os.path.abspath(os.path.dirname(__file__))  
data_folder = os.path.join(base_dir, "../main_data")
cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")  # Updated cleaned file name
data = pd.read_csv(cleaned_csv_file, low_memory=False)

# Ensure the results directory exists
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

# Create a subfolder for bivariate analysis results
bivariate_dir = os.path.join(results_dir, "bivariate")
os.makedirs(bivariate_dir, exist_ok=True)

# Function to save plots with proper naming
def save_plot(fig, plot_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(bivariate_dir, f"{plot_name}_{timestamp}.png")
    fig.savefig(plot_filename)
    print(f"Plot saved as: {plot_filename}")

# Bivariate or Multivariate Analysis
# 1. Scatter Plot for TotalPremium vs TotalClaims by PostalCode
plt.figure(figsize=(12, 8))
scatter_plot = sns.scatterplot(data=data, x="TotalPremium", y="TotalClaims", hue="PostalCode", palette="Set1", alpha=0.7)
scatter_plot.set_title("Scatter Plot of TotalPremium vs TotalClaims by PostalCode")
scatter_plot.set_xlabel("TotalPremium")
scatter_plot.set_ylabel("TotalClaims")

# Manually set the legend's location (e.g., upper left)
scatter_plot.legend(loc='upper left', bbox_to_anchor=(1, 1), title="PostalCode")

save_plot(plt, "scatter_TotalPremium_vs_TotalClaims_by_PostalCode")
plt.close()

# 2. Correlation Matrix (for numerical variables, including TotalPremium and TotalClaims)
corr_cols = ['TotalPremium', 'TotalClaims']  # You can add other columns for correlation as needed
corr_matrix = data[corr_cols].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Matrix for TotalPremium and TotalClaims")
save_plot(plt, "correlation_matrix_TotalPremium_TotalClaims")
plt.close()