import pandas as pd
import sys
import os

if len(sys.argv) < 2:
    print("Usage: python process_table.py <input_csv_file>")
    print("Example: python process_table.py tableA.csv")
    sys.exit(1)

input_file = sys.argv[1]

if not os.path.exists(input_file):
    print(f"Error: File '{input_file}' not found!")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(input_file))[0]
output_file = f"{base_name}_processed.csv"

df = pd.read_csv(input_file)

df['description'] = (
    df['category'].fillna('').astype(str) + ' ' +
    df['brand'].fillna('').astype(str) + ' ' +
    df['modelno'].fillna('').astype(str)
).str.strip()

df = df.drop(columns=['category', 'brand', 'modelno'])

cols = df.columns.tolist()
price_idx = cols.index('price')
cols.remove('description')
cols.insert(price_idx, 'description')
df = df[cols]

df.to_csv(output_file, index=False)

print(f"Processing complete! File saved as {output_file}")
print(f"Total rows processed: {len(df)}")
