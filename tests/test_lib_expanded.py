import pandas as pd
import tyme
import json

def main():
    print("Testing expanded tyme library API...")
    
    # Create simple dataframe
    df = pd.DataFrame({
        "A": [1, 2, 3, 4, 5],
        "B": ["x", "y", "x", "y", "x"]
    })
    
    # 1. Test get_profile
    print("\n1. Testing get_profile(df)...")
    profile = tyme.get_profile(df)
    print(f"Profile keys: {list(profile.keys())}")
    print(f"Columns found: {len(profile['columns'])}")
    
    # 2. Test get_suggestions
    print("\n2. Calling tyme.get_suggestions(df)...")
    suggestions = tyme.get_suggestions(df, model="llama3.2")
    print(f"Received {len(suggestions)} suggestions")

    # 3. Test ask_question
    print("\n3. Testing ask_question()...")
    history = []
    question = "How do I implement the first suggestion?"
    print(f"Question: {question}")
    
    ans = tyme.ask_question(
        profile=profile,
        suggestions=suggestions,
        history=history,
        question=question,
        model="llama3.2"
    )
    print(f"Answer length: {len(ans)}")
    print(f"Answer snippet: {ans[:100]}...")

if __name__ == "__main__":
    main()
