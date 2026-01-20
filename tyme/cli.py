from __future__ import annotations
import argparse
import json
import os
import sys
import time

from .csv_loader import load_csv
from .profile import profile_df
from .prompts import build_suggest_prompt, build_chat_prompt
from .ollama_client import generate_text
from .parsing import parse_suggestions, Suggestion
from .session import SessionState


def _print_suggestions(suggestions: list[Suggestion], limit: int = 10) -> None:
    print(f"\nTop {limit} suggestions:")
    print("=" * 60)
    for i, s in enumerate(suggestions[:limit], start=1):
        print(f"\nSuggestion {i}: {s.name}")
        print(f"  Type: {s.feature_type} | Risk: {s.risk}")
        print("-" * 60)
        print(f"  Why: {s.why.strip()}")
        print(f"  How: {s.how.strip()}")
        print("=" * 60)


def run_command(args: argparse.Namespace) -> int:
    df = load_csv(args.csv_path)
    prof = profile_df(df)

    task = args.task
    target = args.target

    print(f"Loaded: {args.csv_path} ({prof['shape']['rows']} rows, {prof['shape']['cols']} cols)")
    if target:
        print(f"Target: {target}")
    print(f"Model: {args.model}\n")

    # Parse excluded columns
    exclude_cols = [c.strip() for c in args.exclude.split(",")] if args.exclude else []

    # 1) Suggest phase
    suggest_prompt = build_suggest_prompt(prof, task=task, target=target, exclude_columns=exclude_cols)
    raw = generate_text(model=args.model, prompt=suggest_prompt, temperature=0.3, num_predict=1100)

    suggestions = parse_suggestions(raw)
    _print_suggestions(suggestions, limit=args.limit)

    # save initial session (optional)
    session = SessionState(
        csv_path=args.csv_path,
        model=args.model,
        task=task,
        target=target,
        profile=prof,
        suggestions=suggestions,
        history=[],
    )

    # 2) Chat phase
    print("\nChat mode: ask questions about the suggestions. Type 'export' to save, 'exit' to quit.")
    while True:
        try:
            user_in = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_in:
            continue
        if user_in.lower() in {"exit", "quit"}:
            break
        if user_in.lower() == "export":
            # Generate content
            lines = []
            lines.append(f"Top {len(session.suggestions)} suggestions:")
            lines.append("=" * 60)
            for i, s in enumerate(session.suggestions, start=1):
                lines.append(f"\nSuggestion {i}: {s.name}")
                lines.append(f"  Type: {s.feature_type} | Risk: {s.risk}")
                lines.append("-" * 60)
                lines.append(f"  Why: {s.why.strip()}")
                lines.append(f"  How: {s.how.strip()}")
                lines.append("=" * 60)

            lines.append("\n\nChat History:")
            lines.append("=" * 60)
            for msg in session.history:
                role = msg["role"].upper()
                content = msg["content"]
                lines.append(f"\n[{role}]\n{content}")
                lines.append("-" * 40)
            
            # Write to file
            timestamp = int(time.time())
            filename = f"tyme_export_{timestamp}.txt"
            os.makedirs("example", exist_ok=True)
            path = os.path.join("example", filename)
            
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
                print(f"\nSuccessfully exported session to: {path}")
            except Exception as e:
                print(f"\nFailed to export session: {e}")
            
            continue

        # If user types just a number, expand it
        if user_in.isdigit():
            idx = int(user_in)
            if 1 <= idx <= len(session.suggestions):
                s = session.suggestions[idx - 1]
                user_msg = (
                    f"Explain suggestion #{idx} in detail and give a short pandas/sklearn implementation plan.\n"
                    f"Suggestion object: {s.model_dump()}"
                )
            else:
                user_msg = f"The user entered {idx} but it's out of range. Ask them to pick 1..{len(session.suggestions)}."
        else:
            user_msg = user_in

        session.history.append({"role": "user", "content": user_in})

        suggestions_jsonable = [s.model_dump() for s in session.suggestions]
        chat_prompt = build_chat_prompt(
            profile=session.profile,
            suggestions_jsonable=suggestions_jsonable,
            history=session.history,
            user_message=user_msg,
        )

        ans = generate_text(model=session.model, prompt=chat_prompt, temperature=0.4, num_predict=900).strip()
        print(f"\nAssistant: {ans}\n")

        session.history.append({"role": "assistant", "content": ans})

    # optional save
    if args.save:
        payload = {
            "csv_path": session.csv_path,
            "model": session.model,
            "task": session.task,
            "target": session.target,
            "profile": session.profile,
            "suggestions": [s.model_dump() for s in session.suggestions],
            "history": session.history,
        }
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"Saved session to: {args.save}")

    return 0


def main() -> None:
    p = argparse.ArgumentParser(prog="tyme-fe", description="CSV -> profiling -> LLM suggestions -> chat")
    sub = p.add_subparsers(dest="cmd", required=True)

    runp = sub.add_parser("run", help="Generate suggestions then start chat")
    runp.add_argument("csv_path", help="Path to CSV file")
    runp.add_argument("--model", default="llama3.2", help="Ollama model name (e.g. llama3.2, gemma3)")
    runp.add_argument("--target", default=None, help="Target column name (optional)")
    runp.add_argument("--task", default="unspecified", choices=["classification", "regression", "unspecified"], help="Task type")
    runp.add_argument("--limit", type=int, default=10, help="How many suggestions to print initially")
    runp.add_argument("--exclude", default=None, help="Comma-separated list of columns to exclude from suggestions")
    runp.add_argument("--save", default=None, help="Save session JSON to a file path")
    runp.set_defaults(func=run_command)

    args = p.parse_args()
    rc = args.func(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
