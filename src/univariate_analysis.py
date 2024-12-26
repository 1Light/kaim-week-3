import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the cleaned data
base_dir = os.path.abspath(os.path.dirname(__file__))  
data_folder = os.path.join(base_dir, "../main_data")
cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
data = pd.read_csv(cleaned_csv_file, low_memory=False)

# Ensure the results directory exists
results_dir = os.path.join(base_dir, "../results")
os.makedirs(results_dir, exist_ok=True)

# Create a subfolder for univariate results
univariate_dir = os.path.join(results_dir, "univariate")
os.makedirs(univariate_dir, exist_ok=True)

# Function to save plots with proper naming
def save_plot(fig, plot_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(univariate_dir, f"{plot_name}_{timestamp}.png")  # Save in the univariate subfolder
    fig.savefig(plot_filename)
    print(f"Plot saved as: {plot_filename}")

# Univariate Analysis
# 1. Distribution of Numerical Variables (Histogram)
numerical_cols = data.select_dtypes(include=['number']).columns
for col in numerical_cols:
    plt.figure(figsize=(10, 6))
    sns.histplot(data[col], kde=True, bins=30, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Frequency')
    save_plot(plt, f"histogram_{col}")
    plt.close()

# 2. Distribution of Categorical Variables (Bar Chart)
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=col)  # Removed the palette argument
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel('Count')
    save_plot(plt, f"bar_chart_{col}")
    plt.close()
