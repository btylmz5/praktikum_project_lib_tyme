import pandas as pd
import tyme
import json

def main():
    print("Loading data...")
    df = pd.read_csv("data/combined_oulad.csv")
    print(f"Loaded {df.shape}")
    
    print("Generating profile...")
    prof = tyme.get_profile(df)
    
    print("NUMERIC COLUMNS:")
    for col_info in prof["columns"]:
        if col_info["inferred_type"] == "numeric":
            print(f"  - {col_info['name']}")

if __name__ == "__main__":
    main()
