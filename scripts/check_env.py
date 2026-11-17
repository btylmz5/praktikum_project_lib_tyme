import sys
import os

print("Python executable:", sys.executable)
print("CWD:", os.getcwd())

try:
    import pandas as pd
    print("Pandas imported:", pd.__version__)
except ImportError as e:
    print("Pandas import failed:", e)

try:
    import ollama
    print("Ollama imported")
except ImportError as e:
    print("Ollama import failed:", e)

try:
    import tyme.cli
    print("tyme.cli imported")
except ImportError as e:
    print("tyme.cli import failed:", e)

print("Attempting to connect to Ollama...")
try:
    # Just list models to see if it responds
    models = ollama.list()
    print("Ollama models:", models)
except Exception as e:
    print("Ollama connection failed:", e)

print("Done.")
