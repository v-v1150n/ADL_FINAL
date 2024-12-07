import pandas as pd

# List of input file names
input_files = ["result1.csv", "result2.csv", "result3.csv", "result4.csv"]

# Initialize an empty list to hold dataframes
dataframes = []

# Read each file and append the dataframe to the list
for file in input_files:
    df = pd.read_csv(file, encoding="utf-8")
    dataframes.append(df)

# Concatenate all dataframes
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged dataframe to a new CSV file
output_file = "result.csv"
merged_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"Merged {len(input_files)} files into {output_file}")
