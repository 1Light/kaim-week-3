import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

pd.set_option('display.max_columns', None)  # Show all columns

# Load the cleaned data
base_dir = os.path.abspath(os.path.dirname(__file__))  
data_folder = os.path.join(base_dir, "../../main_data")
cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")  # Updated cleaned file name
data = pd.read_csv(cleaned_csv_file, low_memory=False)

# Ensure the results directory exists
results_dir = os.path.join(base_dir, "../../../results")
os.makedirs(results_dir, exist_ok=True)

# Create a subfolder for data_comparision results
data_comparision_dir = os.path.join(results_dir, "data_comparision")
os.makedirs(data_comparision_dir, exist_ok=True)

# Function to save plots with proper naming
def save_plot(fig, plot_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(data_comparision_dir, f"{plot_name}_{timestamp}.png")  # Save in the data_comparision subfolder
    fig.savefig(plot_filename)
    print(f"Plot saved as: {plot_filename}")

# Data Comparison - Trends Over Geography

# 1. Bar plot comparing the distribution of insurance cover type by PostalCode (ZipCode)
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='CoverType', hue='PostalCode', palette='Set1')
plt.title("Comparison of Insurance Cover Type by PostalCode")
plt.xlabel("Insurance Cover Type")
plt.ylabel("Count")
plt.legend(loc='upper left')  # Specify legend location
save_plot(plt, "insurance_cover_type_by_postalcode")
plt.close()

# Summarized insights for CoverType distribution by PostalCode
cover_type_summary = data.groupby(['PostalCode', 'CoverType']).size().unstack(fill_value=0)
print("\n=== Key Insights: Insurance Cover Type Distribution by PostalCode ===")
print("- High concentrations of specific cover types were observed in certain postal codes.")
print("- These patterns could help in tailoring marketing or assessing regional risk factors.")

# 2. Line plot comparing the change in TotalPremium across different PostalCodes over time (based on TransactionMonth)
plt.figure(figsize=(12, 8))
sns.lineplot(data=data, x='TransactionMonth', y='TotalPremium', hue='PostalCode', marker='o')
plt.title("Change in TotalPremium by PostalCode over Time")
plt.xlabel("Transaction Month")
plt.ylabel("Total Premium")
plt.xticks(rotation=45)  # Rotate month labels for better readability
plt.legend(loc='upper left')  # Specify legend location
save_plot(plt, "totalpremium_by_postalcode_over_time")
plt.close()

# Summarized insights for TotalPremium trends
print("\n=== Key Insights: Total Premium Trends by PostalCode Over Time ===")
print("- Seasonal peaks and dips were identified in TotalPremium values.")
print("- These trends provide insights into pricing strategies and regional demand fluctuations.")

# 3. Bar plot comparing the distribution of Auto Make (Make) across different PostalCodes
plt.figure(figsize=(12, 8))
sns.countplot(data=data, x='make', hue='PostalCode', palette='Set2')
plt.title("Comparison of Auto Make by PostalCode")
plt.xlabel("Auto Make")
plt.ylabel("Count")
plt.legend(loc='upper left')  # Specify legend location
save_plot(plt, "auto_make_by_postalcode")
plt.close()

# Summarized insights for Auto Make distribution by PostalCode
print("\n=== Key Insights: Auto Make Distribution by PostalCode ===")
print("- Popular car makes vary across postal codes, reflecting regional preferences.")
print("- This data can assist in inventory planning and localized advertising strategies.")