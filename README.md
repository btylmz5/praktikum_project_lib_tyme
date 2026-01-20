# Tyme

**Tyme** is a powerful CLI tool that uses Large Language Models (LLMs) via [Ollama](https://ollama.ai/) to analyze your CSV datasets. It profiles your data, generates creative feature engineering suggestions, and allows you to chat interactively about your dataset to refine your machine learning strategy.

## Features

- **📊 Data Profiling**: Automatically scans your CSV to understand distributions, missing values, and data types.
- **💡 AI-Powered Suggestions**: Uses local LLMs (like Llama 3) to propose 8-12 actionable feature engineering ideas tailored to your specific data.
- **💬 Interactive Chat**: Discuss the suggestions, ask for implementation details (Pandas/Sklearn code), or helpful explanations directly from the CLI.
- **🔒 Privacy-First**: All processing happens locally with Ollama; your data never leaves your machine.
- **🐍 Python API**: Use `tyme` directly in your Python scripts or Notebooks to analyze DataFrames without saving to CSV.

## Prerequisites

1. **Python 3.10+** installed.
2. **Ollama** installed and running.
   - Install from [ollama.ai](https://ollama.ai).
   - Pull a model (e.g., `llama3.2`):
     ```bash
     ollama pull llama3.2
     ```

## Installation

Clone the repository and install the package (recommended in a virtual environment):

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in editable mode
pip install -e .
```

## Usage

Run the tool by pointing it to a CSV file. You must have the Ollama server running in the background.

```bash
python -m tyme.cli run <path_to_csv> [options]
```

### Examples

**Basic run with default model:**
```bash
python -m tyme.cli run data/combined_oulad.csv
```

**Specify a different model:**
```bash
python -m tyme.cli run data/my_data.csv --model mistral
```

**Target a specific column for prediction:**
```bash
python -m tyme.cli run data/house_prices.csv --target Price --task regression
```

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `files` | Path to the CSV file (required positional argument). | N/A |
| `--model` | Name of the Ollama model to use. | `llama3.2` |
| `--target` | Name of the target column you want to predict. | `None` |
| `--task` | Type of ML task: `classification`, `regression`, or `unspecified`. | `unspecified` |
| `--limit` | Number of top suggestions to display initially. | `10` |
| `--exclude` | Comma-separated list of columns to exclude from suggestions. | `None` |
| `--save` | Path to save the session history and suggestions as a JSON file. | `None` |

## Workflow

1. **Analyze**: Tyme loads your CSV and creates a statistical profile (without sending the full dataset to the LLM).
2. **Suggest**: It prompts the LLM with the profile to generate feature engineering ideas.
3. **Chat**: You enter an interactive session.
   - Type a suggestion number (e.g., `1`) to get detailed implementation steps.
   - Ask general questions like *"How do I handle the missing values in column X?"*.
   - Type `export` to save the suggestions and chat history to a text file in `example/`.
   - Type `exit` or `quit` to leave.

## Library Integration

## Library Integration

To use `tyme` in another project, you first need to install it.

1.  **Install the package**:
    Run this command from your other project's environment, pointing to the `tyme` folder:
    ```bash
    pip install -e /path/to/praktikum_project_lib_tyme
    ```

2.  **Import and use**:
    Now you can import `tyme` in your Python scripts:

```python
import pandas as pd
import tyme

# Load and preprocess your data
df = pd.read_csv("data.csv")
# ... your cleaning steps ...

# Get suggestions directly
suggestions = tyme.get_suggestions(df, model="llama3.2")

for s in suggestions:
    print(f"{s.name}: {s.why}")
```

### API Reference

#### `tyme.get_suggestions(df, model="llama3.2", task="unspecified", target=None, exclude_columns=None)`

Analyze a pandas DataFrame and return a list of feature engineering suggestions.

**Arguments:**

- `df` (pd.DataFrame): The input pandas DataFrame.
- `model` (str): Name of the Ollama model to use (default: `"llama3.2"`).
- `task` (str): The machine learning task type. Options: `"classification"`, `"regression"`, `"unspecified"` (default).
- `target` (str | None): The name of the target column (optional).
- `exclude_columns` (list[str] | None): A list of column names to exclude from suggestions (e.g., IDs, leakage columns).

**Returns:**

- `list[Suggestion]`: A list of objects with the following attributes:
    - `name` (str): Name of the proposed feature.
    - `feature_type` (str): Type of the feature (e.g., "numeric", "categorical").
    - `risk` (str): Potential risk (e.g., "leakage", "none").
    - `why` (str): Explanation of why this feature is useful.
    - `how` (str): Description or pseudocode of how to implement it.
