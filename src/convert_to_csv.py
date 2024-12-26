import os
import pandas as pd

# Define absolute paths
base_dir = os.path.abspath(os.path.dirname(__file__))  
data_folder = os.path.join(base_dir, "../")        
input_file = os.path.join(data_folder, "data/ml.txt")       
main_data_folder = os.path.join(data_folder, "main_data")  
output_file = os.path.join(main_data_folder, "ml.csv") 

# Ensure the main_data folder exists
os.makedirs(main_data_folder, exist_ok=True)

try:
    print(f"Loading data from {input_file}...")
    # Read the pipe-separated file
    data = pd.read_csv(input_file, delimiter="|")
    print("Data loaded successfully!")

    # Convert to CSV and save in the main_data folder
    print(f"Converting to CSV and saving to {output_file}...")
    data.to_csv(output_file, index=False)
    print(f"File successfully converted and stored in {main_data_folder}!")

except FileNotFoundError:
    print(f"Error: The file {input_file} does not exist. Please ensure the path is correct.")
except Exception as e:
    print(f"An error occurred: {e}")
