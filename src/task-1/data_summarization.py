import os
import pandas as pd

# Define file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
main_data_dir = os.path.join(current_dir, "../main_data")
results_dir = os.path.join(current_dir, "../results")
data_file = os.path.join(main_data_dir, "cleaned_ml.csv")

# Ensure results directory exists
os.makedirs(results_dir, exist_ok=True)

# Create a subfolder for data_summarization results
data_summarization_dir = os.path.join(results_dir, "data_summarization")
os.makedirs(data_summarization_dir, exist_ok=True)

# Load the data
print(f"Loading data from {data_file}...")
try:
    data = pd.read_csv(data_file)
    print("Data loaded successfully!")
except FileNotFoundError:
    print(f"Error: File not found at {data_file}")
    exit()

# Descriptive Statistics for Numerical Features
numerical_features = data.select_dtypes(include=["number"])
descriptive_stats = numerical_features.describe()

print("\nDescriptive Statistics for Numerical Features:")
print(descriptive_stats)

# Variability (Standard Deviation) for Numerical Features
variability = numerical_features.std()
print("\nVariability (Standard Deviation) for Numerical Features:")
print(variability)

# Data Structure and Dtypes
data_structure = data.dtypes
print("\nData Structure and Dtypes:")
print(data_structure)

# Save summary to the data_summarization subfolder
output_file_summary = os.path.join(data_summarization_dir, "data_summary.csv")
descriptive_stats.to_csv(output_file_summary)
print(f"\nDescriptive statistics summary saved to {output_file_summary}.")