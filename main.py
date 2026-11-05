import json
import sys
import pandas as pd
from tabulate import tabulate

from agents.cleaner import (
    find_numeric_like_columns,
    clean_dataset,
)
from agents.cleaning_planner import create_cleaning_plan
from agents.analyst import analyze_dataset, answer_question
from agents.validator import validate_cleaning_plan
from agents.reporter import display_dataset_report


def dataset_checks(df: pd.DataFrame) -> dict:
    """Return a small, comparable summary before and after cleaning."""
    return {
        "row_count": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }


def main():
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"E:\Data_Analysis_Agent\data\AI_Usage_and_Impact_on_Students_and_Professionals.csv"
    )

    # 1. Load the original dataset
    df = pd.read_csv(csv_path)
    before_checks = dataset_checks(df)
    print("\n--- Original Dataset ---")
    print(df.head())

    # 2. Create the dataset profile and detect numeric-like columns
    profile = {
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().to_dict(),
        "sample_values": {
            column: df[column].dropna().astype(str).head(5).tolist()
            for column in df.columns
        },
    }
    numeric_candidates = find_numeric_like_columns(df)

    print("\n--- Numeric-Like Columns ---")
    print(json.dumps(numeric_candidates, indent=2, ensure_ascii=False))

    # 3. Count exact duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # 4. Ask Groq to propose a cleaning plan
    plan = create_cleaning_plan(
        profile=profile,
        numeric_candidates=numeric_candidates,
        duplicate_rows=duplicate_rows,
    )

    actions = plan.get("actions", [])
    print("\n--- LLM Cleaning Plan ---")
    rows = [
        [
            item.get("action", ""),
            item.get("column", "—"),
            item.get("strategy", "—"),
            item.get("reason", ""),
        ]
        for item in actions
    ]
    print(tabulate(
        rows,
        headers=["Action", "Column", "Strategy", "Reason"],
        tablefmt="github",
    ))

    # 5. Validate the plan
    validation = validate_cleaning_plan(plan, df.columns.tolist())

    print("\n--- Valid Actions ---")
    print(json.dumps(validation["valid_actions"], indent=2))

    print("\n--- Rejected Actions ---")
    print(json.dumps(validation["rejected_actions"], indent=2))

    # 6. Execute the accepted actions
    cleaned_df, cleaning_log = clean_dataset(
        df,
        validation["valid_actions"],
    )

    # 7. Save the cleaned dataset and log
    output_path = r"E:\Data_Analysis_Agent\data\cleaned.csv"
    cleaned_df.to_csv(output_path, index=False)
    after_checks = dataset_checks(cleaned_df)

    print(f"\nCleaned dataset saved to: {output_path}")
    print(f"Rows before cleaning: {len(df)}")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print("\n--- Cleaning Log ---")
    print(json.dumps(cleaning_log, indent=2, ensure_ascii=False))
    print("\n--- Before/After Checks ---")
    print(json.dumps({"before": before_checks, "after": after_checks}, indent=2))

    # 8. Display Dataset Comparison Report (Original df vs Cleaned df)
    display_dataset_report(df, cleaned_df)

    # 9. Analyze the cleaned DataFrame and optionally answer one question.
    print("\n--- Dataset Analysis ---")
    print(json.dumps(analyze_dataset(cleaned_df), indent=2, ensure_ascii=False))

    try:
        question = input("\nAsk a question about the cleaned dataset (or press Enter to skip): ")
    except EOFError:
        question = ""

    if question.strip():
        answer = answer_question(cleaned_df, question)
        print("\n--- Analysis Answer ---")
        print(answer["answer"])
        if "result" in answer:
            print(json.dumps(answer["result"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
