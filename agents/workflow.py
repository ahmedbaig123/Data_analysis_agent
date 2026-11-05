from typing import Any, TypedDict
import json
import sys

import pandas as pd
from langgraph.graph import StateGraph, START, END

from agents.cleaner import find_numeric_like_columns, clean_dataset
from agents.cleaning_planner import create_cleaning_plan
from agents.validator import validate_cleaning_plan
from agents.reporter import display_dataset_report
from agents.analyst import analyze_dataset, answer_question

class WorkflowState(TypedDict):
    csv_path: str
    question: str

    df: pd.DataFrame
    profile: dict[str, Any]
    numeric_candidates: list[Any]
    duplicate_rows: int

    plan: dict[str, Any]
    validation: dict[str, Any]

    cleaned_df: pd.DataFrame
    report_metrics: dict[str, Any]
    cleaning_log: list[Any]
    analysis: dict[str, Any]
    answer: dict[str, Any]

def load_dataset(state: WorkflowState) -> dict:
     df = pd.read_csv(state["csv_path"])
     return {"df": df}

def profile_dataset(state: WorkflowState) -> dict:
    df = state["df"]

    profile = {
        "columns":df.columns.tolist(),
        "dtypes":df.dtypes.astype(str).to_dict(),
        "missing_values":df.isna().sum().to_dict(),
        "sample_values": {
            column: df[column].dropna().astype(str).head(5).tolist()
            for column in df.columns
        },
    }

    return {"profile": profile,"numeric_candidates": find_numeric_like_columns(df),"duplicate_rows":int(df.duplicated().sum())}

def plan_cleaning(state: WorkflowState) -> dict:
    plan = create_cleaning_plan(
        profile = state["profile"],
        numeric_candidates = state["numeric_candidates"],
        duplicate_rows = state["duplicate_rows"],
    )

    return {"plan":plan}

def validate_plan(state: WorkflowState) -> dict:
    validation = validate_cleaning_plan(
        state["plan"],
        state["df"].columns.tolist(),
    )
    return {"validation": validation}

def clean_dataset_node(state: WorkflowState) -> dict:
    cleaned_df,cleaning_log = clean_dataset(
        state["df"],
        state["validation"]["valid_actions"],
    )
    return {
        "cleaned_df": cleaned_df,
        "cleaning_log": cleaning_log,
    }

def generate_report(state: WorkflowState) -> dict:
    report_metrics = display_dataset_report(
        state["df"],
        state["cleaned_df"],
    )
    return {"report_metrics": report_metrics}

def analyze_cleaned_dataset(state: WorkflowState) -> dict:
    return {"analysis": analyze_dataset(state["cleaned_df"])}

def answer_user_question(state: WorkflowState) -> dict:
    question = state.get("question", "").strip()
    if not question:
        return {"answer": {}}

    return {
        "answer": answer_question(state["cleaned_df"], question)
    }

builder = StateGraph(WorkflowState)

builder.add_node("load_dataset", load_dataset)
builder.add_node("profile_dataset", profile_dataset)
builder.add_node("plan_cleaning", plan_cleaning)
builder.add_node("validate_plan", validate_plan)
builder.add_node("clean_dataset", clean_dataset_node)
builder.add_node("generate_report", generate_report)
builder.add_node("analyze_cleaned_dataset", analyze_cleaned_dataset)
builder.add_node("answer_user_question", answer_user_question)

builder.add_edge(START, "load_dataset")
builder.add_edge("load_dataset", "profile_dataset")
builder.add_edge("profile_dataset", "plan_cleaning")
builder.add_edge("plan_cleaning", "validate_plan")
builder.add_edge("validate_plan", "clean_dataset")
builder.add_edge("clean_dataset", "generate_report")
builder.add_edge("generate_report", "analyze_cleaned_dataset")
builder.add_edge("analyze_cleaned_dataset", "answer_user_question")
builder.add_edge("answer_user_question", END)

workflow = builder.compile()

if __name__ == "__main__":
    csv_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else r"E:\Data_Analysis_Agent\data\AI_Usage_and_Impact_on_Students_and_Professionals.csv"
    )

    question = input(
        "Ask a question about the cleaned dataset "
        "(or press Enter to skip): "
    )

    result = workflow.invoke({
        "csv_path": csv_path,
        "question": question,
    })

    print("\n--- Cleaning Log ---")
    print(json.dumps(result["cleaning_log"], indent=2, ensure_ascii=False))

    print("\n--- Dataset Analysis ---")
    print(json.dumps(result["analysis"], indent=2, ensure_ascii=False))

    if result.get("answer"):
        print("\n--- Analysis Answer ---")
        print(result["answer"].get("answer", ""))
        if "result" in result["answer"]:
            print(json.dumps(
                result["answer"]["result"],
                indent=2,
                ensure_ascii=False,
            ))