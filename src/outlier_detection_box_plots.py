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

# Create a subfolder for outlier detection results
outlier_dir = os.path.join(results_dir, "outlier")
os.makedirs(outlier_dir, exist_ok=True)

# Function to save plots with proper naming
def save_plot(fig, plot_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(outlier_dir, f"{plot_name}_{timestamp}.png")  # Save in the outlier subfolder
    fig.savefig(plot_filename)
    print(f"Plot saved as: {plot_filename}")

# Outlier Detection - Box Plots for Numerical Columns
# List of numerical columns in the dataset (adjust if necessary)
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns

# Create box plots for each numerical column to detect outliers
for col in numerical_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x=col)  # Removed the palette argument
    plt.title(f"Box Plot of {col} - Outlier Detection")
    plt.xlabel(col)
    save_plot(plt, f"outlier_detection_{col}")
    plt.close()