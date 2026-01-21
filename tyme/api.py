from __future__ import annotations
import pandas as pd
from typing import Any, Optional, Literal, Mapping

from .profile import profile_df
from .prompts import build_suggest_prompt, build_chat_prompt
from .ollama_client import generate_text
from .parsing import parse_suggestions, Suggestion

from ollama._types import Options
from ollama import chat

def get_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Generate a statistical profile of the DataFrame.
    
    Args:
        df: Input pandas DataFrame.
        
    Returns:
        Dictionary containing profile metadata (shapes, columns, types, stats).
    """
    return profile_df(df)

def ask_question(
    profile: dict[str, Any],
    suggestions: list[Suggestion],
    history: list[dict[str, str]],
    question: str,
    model: str = "llama3.2"
) -> str:
    """
    Ask a question about the dataset/suggestions in a chat context.

    Args:
        profile: Dataset profile (from get_profile).
        suggestions: List of Suggestion objects (from get_suggestions).
        history: List of chat messages (role/content dicts).
        question: The user's question.
        model: Ollama model name.

    Returns:
        The LLM's answer as a string.
    """
    suggestions_jsonable = [s.model_dump() for s in suggestions]
    
    chat_prompt = build_chat_prompt(
        profile=profile,
        suggestions_jsonable=suggestions_jsonable,
        history=history,
        user_message=question
    )

    ans = generate_text(
        model=model, 
        prompt=chat_prompt, 
        temperature=0.4, 
        num_predict=900
    )
    return ans.strip()

def get_suggestions(
    df: pd.DataFrame,
    model: str = "llama3.2",
    task: str = "unspecified",
    target: Optional[str] = None,
    exclude_columns: Optional[list[str]] = None,
) -> list[Suggestion]:
    """
    Analyze a DataFrame and generate feature engineering suggestions using an LLM.

    Args:
        df: Input pandas DataFrame.
        model: Ollama model name (default: "llama3.2").
        task: ML task type ("classification", "regression", "unspecified").
        target: Target column name (optional).
        exclude_columns: List of columns to exclude from suggestions.

    Returns:
        List of Suggestion objects.
    """
    # 1. Profile the DataFrame
    prof = profile_df(df)

    # 2. Build the prompt
    suggest_prompt = build_suggest_prompt(
        prof, 
        task=task, 
        target=target, 
        exclude_columns=exclude_columns
    )

    # 3. Call LLM
    raw = generate_text(
        model=model, 
        prompt=suggest_prompt, 
        temperature=0.3, 
        num_predict=2500
    )

    # 4. Parse response
    suggestions = parse_suggestions(raw)
    return suggestions

def chat_continuous(
        initial_prompt : str = None,
        prior_messages : list = [], 
        quit_message : str = "Quit", 
        model : str = "llama3.2",
        think : bool | Literal['low', 'medium', 'high'] | None = None,
        options : Mapping[str, Any] | Options | None = None
        ):
    
    message_history = prior_messages

    while True:
        if not initial_prompt is None:
            input1 = handle_user_input(initial_prompt)
            print(input1)
            initial_prompt = None
        else:
            input1 = handle_user_input()
        if input1 == quit_message:
            print("\n\nYou have quit the chat.\n\n")
            break
        message_history.append({
            'role': 'user',
            'content': input1
        })
        response = chat(
            model=model, 
            messages=message_history, 
            think=think,
            options=options
            )
        message_history.append(response.message)
        print("\n" + model + ":\n")
        print(response.message.content)


def get_suggestions_and_chat_continuous(
        df : pd.DataFrame,
        row_amount : int = 50,
        model_type : str | None = None,
        model_algorithm : str | None = None, 
        target_variable : str | None = None, 
        classes : str | None = None, 
        additional_information : str | None = None,
        custom_suggestion_structure : str | None = None,
        quit_messsage : str = "Quit",
        model : str = "deepseek-r1",
        think: bool | Literal['low', 'medium', 'high'] | None = True,
        options : Mapping[str, Any] | Options | None = None
        ):
    
    if row_amount > len(df):
        row_amount = len(df)
        print("\nWarning: n is greater than the number of rows. n = " + str(row_amount) + ", length = " + str(len(df)) + ".\n\n")
    
    initial_prompt = "You are going to help engineer features for a predictive AI model."
    if model_type:
        initial_prompt += " The model handles predictions of the following type: " + model_type + "."
    if model_algorithm:
        initial_prompt += " The model uses the following algorithm: " + model_algorithm + "."
    initial_prompt += " You will receive information about the dataset such as the names of the columns and sample observations."
    initial_prompt += " The column names are as follows:\n"
    initial_prompt += ', '.join(df.columns) + "."
    if target_variable:
        initial_prompt += "\nThe target variable is: " + target_variable + "."
    if classes:
        initial_prompt += " The classes are: " + classes + "."
    if additional_information:
        initial_prompt += " Some additional information: " + additional_information + "."
    initial_prompt += "\nSample observations:\n"
    initial_prompt += df.sample(n = row_amount).to_string()
    initial_prompt += "\nIn your answer include only a list of feature suggestions, with a suggestion containing only the following parts: "
    if custom_suggestion_structure:
        initial_prompt += custom_suggestion_structure
    else:
        initial_prompt += "Name of the feature, a moderately long explanation of why it would help, and how the suggestion can be engineered in code."

    chat_continuous(initial_prompt=initial_prompt, quit_message=quit_messsage, think=think, model=model, options=options)


def custom_prompt_chat_with_mem(
        custom_prompt : str, 
        quit_messsage : str = "Quit",
        model : str = "deepseek-r1",
        think: bool | Literal['low', 'medium', 'high'] | None = True,
        options : Mapping[str, Any] | Options | None = None
        ):

    initial_prompt = custom_prompt
    
    chat_continuous(initial_prompt=initial_prompt, quit_message=quit_messsage, think=think, model=model, options=options)


def handle_user_input(user_input : str = None):
    print("\nUser Input:\n")
    if user_input is None:
        user_input = input()
    return user_input
