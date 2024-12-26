import pandas as pd
import os

# Define absolute paths
base_dir = os.path.abspath(os.path.dirname(__file__))  
data_folder = os.path.join(base_dir, "../main_data")   
csv_file = os.path.join(data_folder, "ml.csv")         

# Load the CSV file into a DataFrame
print(f"Loading data from {csv_file}...")
data = pd.read_csv(csv_file, low_memory=False)  # low_memory=False avoids dtype warnings during import
print("Data loaded successfully!")

# Data Quality Assessment:
# Check for missing values
print("\nData Quality Assessment: Checking for missing values...")
missing_data = data.isnull().sum()  # Count missing values per column
missing_percentage = (missing_data / len(data)) * 100  # Percentage of missing values

# Create a DataFrame to show missing values and their percentage
missing_summary = pd.DataFrame({'Missing Values': missing_data, 'Percentage': missing_percentage})
missing_summary = missing_summary[missing_summary['Missing Values'] > 0]  # Only show columns with missing data

if not missing_summary.empty:
    print("\nColumns with Missing Data:")
    print(missing_summary)
else:
    print("\nNo missing values detected.")

# Check for empty columns (columns where all values are NaN)
empty_columns = data.columns[data.isnull().all()]
print(f"\nEmpty Columns (all values are NaN): {list(empty_columns)}")

# Drop the empty columns
data = data.drop(columns=empty_columns)

# Convert 'CapitalOutstanding' to numeric (coercing errors)
data['CapitalOutstanding'] = pd.to_numeric(data['CapitalOutstanding'], errors='coerce')

# Convert 'TransactionMonth' and 'VehicleIntroDate' to datetime
data['TransactionMonth'] = pd.to_datetime(data['TransactionMonth'], errors='coerce')

# Convert 'VehicleIntroDate' to datetime with the correct format
def parse_vehicle_intro_date(date_value):
    # Try parsing the 'Month-Year' format (e.g., "June-02")
    try:
        return pd.to_datetime(data['VehicleIntroDate'], format='%m/%Y', errors='coerce')
    except Exception:
        # If that fails, try parsing general datetime formats (e.g., "2/1/2014 12:00:00 AM")
        return pd.to_datetime(date_value, errors='coerce')

# Convert numeric-like columns to integers where appropriate
integer_columns = ['Cylinders', 'NumberOfDoors']
for col in integer_columns:
    data[col] = pd.to_numeric(data[col], downcast='integer', errors='coerce')

# Convert postal codes to strings
data['PostalCode'] = data['PostalCode'].astype(str)

# Convert boolean-like columns to actual booleans
boolean_columns = [
    'AlarmImmobiliser', 'TrackingDevice', 'CapitalOutstanding', 'NewVehicle', 
    'WrittenOff', 'Rebuilt', 'Converted', 'CrossBorder'
]
for col in boolean_columns:
    # First, ensure the column is treated as a string, then strip and convert to lower case
    data[col] = data[col].astype(str).str.strip().str.lower().map({'yes': True, 'no': False})

# Apply consistent formatting to object columns (capitalize first letter of each word)
for col, dtype in data.dtypes.items():
    if dtype == "object":
        data[col] = data[col].astype(str).str.strip()  # Remove leading/trailing whitespace
        data[col] = data[col].str.capitalize()  # Standardize text (capitalize first letter of each word)

# Fill missing values (if any) for categorical columns with mode (most frequent value) and for numerical columns with median
for col in data.columns:
    if data[col].dtype == 'object':  # For categorical columns
        mode_value = data[col].mode()[0]
        data[col] = data[col].fillna(mode_value)
    elif data[col].dtype in ['float64', 'int64']:  # For numerical columns
        median_value = data[col].median()
        data[col] = data[col].fillna(median_value)

# Optionally, save cleaned data to a new CSV for verification
cleaned_csv_file = os.path.join(data_folder, "cleaned_ml.csv")
data.to_csv(cleaned_csv_file, index=False)
print(f"Cleaned data saved to {cleaned_csv_file}!")