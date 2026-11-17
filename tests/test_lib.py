import pandas as pd
import tyme

def main():
    print("Testing tyme library API...")
    
    # Create simple dataframe
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["x", "y", "x", "y", "x"]
    })
    
    print("Calling tyme.get_suggestions(df)...")
    suggestions = tyme.get_suggestions(df, model="llama3.2")
    
    print(f"\nReceived {len(suggestions)} suggestions:")
    for i, s in enumerate(suggestions, 1):
        print(f"{i}. {s.name} ({s.feature_type}): {s.why[:50]}...")

if __name__ == "__main__":
    main()
