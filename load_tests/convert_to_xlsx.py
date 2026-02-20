import pandas as pd
import glob

# Find all CSV files in reports folder
csv_files = glob.glob("reports/*.csv")

for csv_file in csv_files:
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Create XLSX filename
    xlsx_file = csv_file.replace('.csv', '.xlsx')
    
    # Save to Excel
    df.to_excel(xlsx_file, index=False)
    print(f"Converted: {xlsx_file}")