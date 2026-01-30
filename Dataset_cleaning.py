###Data_cleaning.py###
import pandas as pd
import numpy as np

# 1. Load your dataset
df = pd.read_csv('fusion_inhibition_data.csv', low_memory=False)

# 2. Create wt_average - you already have it as 'WT_Inhibition @ 10 uM Avg'!
print("Column names found:")
for col in df.columns:
    print(f"  '{col}'")

# 3. Keep only the columns you want
columns_to_keep = [
    'PUBCHEM_SID',                    # pubchemID
    'PUBCHEM_EXT_DATASOURCE_SMILES',  # smiles
    'PUBCHEM_ACTIVITY_OUTCOME',       # pubchem activity outcome
    'WT_Inhibition @ 10 uM Avg'       # average inhibition
]

df_simple = df[columns_to_keep].copy()

# 4. Rename for clarity
df_simple.columns = ['pubchem_id', 'smiles', 'activity_outcome', 'wt_average']

# 5. Convert wt_average to numeric (just in case)
df_simple['wt_average'] = pd.to_numeric(df_simple['wt_average'], errors='coerce')

# 6. Create Active/Inactive label (1/0)
THRESHOLD = 57.94  # ≥57.94% inhibition = active
df_simple['active'] = (df_simple['wt_average'] >= THRESHOLD).astype(int)

# 7. Remove any rows with missing SMILES or wt_average
df_simple = df_simple.dropna(subset=['smiles', 'wt_average'])

# 8. Save the clean dataset
df_simple.to_csv('clean_dataset_simple.csv', index=False)

# 9. Show statistics
print("\n=== CLEANED DATASET ===")
print(f"Total compounds: {len(df_simple)}")
print(f"Active (≥{THRESHOLD}% inhibition): {df_simple['active'].sum()}")
print(f"Inactive: {(df_simple['active'] == 0).sum()}")
print(f"Active percentage: {df_simple['active'].sum() / len(df_simple) * 100:.2f}%")

# Show first few rows
print("\nFirst 5 rows:")
print(df_simple.head())

# Optional: Check agreement with PubChem's labels
print("\n=== ACTIVITY OUTCOME CHECK ===")
print(df_simple['activity_outcome'].value_counts())
