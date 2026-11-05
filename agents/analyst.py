"""Pandas-based dataset summaries and intelligent natural-language answers."""

import os
import re
from dotenv import load_dotenv
import pandas as pd

load_dotenv()


def _display_value(value, limit: int = 100) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit - 3]}..."


def _top_values(series: pd.Series, limit: int = 5) -> list[dict]:
    counts = series.dropna().value_counts().head(limit)
    return [
        {"value": _display_value(value), "count": int(count)}
        for value, count in counts.items()
    ]


def _numeric_summary(series: pd.Series) -> dict:
    values = series.dropna()
    if values.empty:
        return {"count": 0}

    return {
        "count": int(values.count()),
        "mean": float(values.mean()),
        "std": float(values.std()) if len(values) > 1 else None,
        "min": float(values.min()),
        "25%": float(values.quantile(0.25)),
        "median": float(values.median()),
        "75%": float(values.quantile(0.75)),
        "max": float(values.max()),
    }


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Build a general-purpose report from the supplied DataFrame."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [
        column for column in df.columns if column not in numeric_columns
    ]
    missing = df.isna().sum()
    duplicate_rows = int(df.duplicated().sum())

    issues = []
    if duplicate_rows:
        issues.append(f"{duplicate_rows} exact duplicate row(s) found.")
    for column, count in missing[missing > 0].items():
        issues.append(f"{column} has {int(count)} missing value(s).")
    for column in df.columns:
        if df[column].dropna().nunique() == 1:
            issues.append(f"{column} has only one non-missing value.")

    return {
        "overview": {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict(),
        },
        "numeric_summary": {
            column: _numeric_summary(df[column]) for column in numeric_columns
        },
        "categorical_summary": {
            column: {
                "non_missing_count": int(df[column].notna().sum()),
                "unique_count": int(df[column].nunique(dropna=True)),
                "top_values": _top_values(df[column]),
            }
            for column in categorical_columns
        },
        "missing_values": {
            column: int(count) for column, count in missing.items() if count
        },
        "data_quality_issues": issues,
    }


def _normalise_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _find_column(question: str, columns: list[str]) -> str | None:
    """Find the best matching column name from the question."""
    question_norm = _normalise_text(question)
    question_words = f" {question_norm} "

    # 1. Exact normalized phrase match
    matches = [
        col
        for col in columns
        if f" {_normalise_text(col)} " in question_words
    ]
    if matches:
        return max(matches, key=len)

    # 2. Substring match for stripped column names
    for col in sorted(columns, key=len, reverse=True):
        col_norm = _normalise_text(col)
        if col_norm and (f" {col_norm} " in question_words or col_norm == question_norm):
            return col

    # 3. Partial / Token-based matching (e.g. "lung cancer" -> "LUNG_CANCER")
    q_tokens = set(question_norm.split())
    candidates = []
    for col in columns:
        col_tokens = {t for t in _normalise_text(col).split() if len(t) >= 3}
        common = q_tokens.intersection(col_tokens)
        if common:
            candidates.append((len(common), len(col), col))

    if candidates:
        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return candidates[0][2]

    return None


def _resolve_follow_up(
    question: str,
    chat_history: list[dict] | None,
    columns: list[str],
) -> str:
    """Resolve a simple follow-up using the most recent user question."""
    if not chat_history:
        return question

    previous_user_questions = [
        message["content"]
        for message in chat_history
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
    ]
    if not previous_user_questions:
        return question

    normalized = _normalise_text(question)
    follow_up_starts = (
        "what about",
        "how about",
        "and ",
        "also ",
        "what is its",
        "what is that",
        "show me that",
        "compare it",
        "compare them",
    )

    if not normalized.startswith(follow_up_starts):
        return question

    current_column = _find_column(question, columns)
    if current_column:
        return question

    for previous_question in reversed(previous_user_questions):
        previous_column = _find_column(previous_question, columns)
        if previous_column:
            return f"{question} for {previous_column}"

    return question


def _ask_groq_llm(df: pd.DataFrame, question: str, chat_history: list[dict] | None = None) -> dict | None:
    """Use Groq LLM to intelligently answer complex or open-ended questions when online."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        schema = {str(c): str(t) for c, t in df.dtypes.items()}
        sample_rows = df.head(3).to_dict(orient="records")
        context = {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": df.columns.tolist(),
            "data_types": schema,
            "sample_data": sample_rows,
        }

        system_prompt = (
            "You are a helpful, senior data analyst assistant. "
            "Answer the user's question directly, clearly, and concisely based on the dataset summary provided. "
            "Keep the answer friendly and factual."
        )

        user_content = f"Dataset Overview:\n{context}\n\nQuestion: {question}"

        messages = [{"role": "system", "content": system_prompt}]
        if chat_history:
            for msg in chat_history[-4:]:
                if msg.get("role") in ("user", "assistant") and isinstance(msg.get("content"), str):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": user_content})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=450,
        )
        answer_text = response.choices[0].message.content.strip()
        return {
            "answer": answer_text,
            "result": {"status": "answered_by_llm", "question": question},
        }
    except Exception:
        return None


def answer_question(
    df: pd.DataFrame,
    question: str,
    chat_history: list[dict] | None = None,
) -> dict:
    """Answer dataset questions using deterministic pandas results and LLM fallback."""
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question about this dataset."}

    question = _resolve_follow_up(
        question,
        chat_history,
        df.columns.tolist(),
    )
    normalized = _normalise_text(question)

    # 1. Duplicate rows questions (e.g. "were there any duplicates?", "duplicate count")
    if any(w in normalized for w in ("duplicate", "duplicates", "duplicated", "redundant row")):
        dup_count = int(df.duplicated().sum())
        if dup_count == 0:
            return {
                "answer": f"There are no duplicate rows in the dataset (0 duplicates found across {len(df):,} total rows).",
                "result": {"duplicate_rows": 0, "total_rows": len(df)},
            }
        return {
            "answer": f"The dataset contains {dup_count:,} duplicate row(s) out of {len(df):,} total rows.",
            "result": {"duplicate_rows": dup_count, "total_rows": len(df)},
        }

    # 2. Columns / Features / Variables / Attributes
    if any(w in normalized for w in ("feature", "features", "column", "columns", "variable", "variables", "attribute", "attributes", "field", "fields")):
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in cols if c not in num_cols]

        if any(w in normalized for w in ("how many", "count", "number of", "total")):
            return {
                "answer": f"The dataset has {len(cols)} features (columns): {len(num_cols)} numeric and {len(cat_cols)} categorical.",
                "result": {
                    "total_features": len(cols),
                    "numeric_features": num_cols,
                    "categorical_features": cat_cols,
                },
            }
        elif any(w in normalized for w in ("what are", "which are", "list", "names", "show")):
            return {
                "answer": f"The dataset contains {len(cols)} columns: {', '.join(cols)}.",
                "result": {"columns": cols, "count": len(cols)},
            }
        else:
            return {
                "answer": f"The dataset has {len(cols)} columns: {', '.join(cols)}.",
                "result": {"columns": cols, "count": len(cols)},
            }

    # 3. Rows / Records / Observations / Samples
    if any(w in normalized for w in ("row", "rows", "record", "records", "sample", "samples", "entry", "entries", "observation", "observations")):
        return {
            "answer": f"The dataset has {len(df):,} rows (records) and {len(df.columns)} columns.",
            "result": {"row_count": len(df), "column_count": len(df.columns)},
        }

    # 4. Shape / Dimensions / Size
    if any(w in normalized for w in ("shape", "dimension", "dimensions", "size")):
        return {
            "answer": f"The dataset shape is {len(df):,} rows by {len(df.columns)} columns.",
            "result": {"rows": len(df), "columns": len(df.columns)},
        }

    # 5. Missing / Null Values
    if any(w in normalized for w in ("missing", "null", "empty", "nan", "na ")):
        total_missing = int(df.isna().sum().sum())
        missing_by_col = {
            col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0
        }
        if total_missing == 0:
            return {
                "answer": f"There are no missing values in this dataset (all {len(df) * len(df.columns):,} cells have valid data).",
                "result": {"total_missing": 0, "missing_by_column": {}},
            }
        breakdown = ", ".join(f"{c}: {n}" for c, n in missing_by_col.items())
        return {
            "answer": f"The dataset has {total_missing:,} missing values across {len(missing_by_col)} column(s): {breakdown}.",
            "result": {"total_missing": total_missing, "missing_by_column": missing_by_col},
        }

    # 6. High-level Overview / Summary
    if any(w in normalized for w in ("overview", "summary", "summarize", "describe the dataset", "about the dataset", "what is this data")):
        num_cols = df.select_dtypes(include="number").columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]
        return {
            "answer": (
                f"This dataset contains {len(df):,} rows and {len(df.columns)} columns. "
                f"It includes {len(num_cols)} numeric column(s) and {len(cat_cols)} categorical column(s). "
                f"Columns include: {', '.join(df.columns)}."
            ),
            "result": {
                "rows": len(df),
                "columns": df.columns.tolist(),
                "numeric_columns": num_cols,
                "categorical_columns": cat_cols,
                "duplicates": int(df.duplicated().sum()),
                "total_missing": int(df.isna().sum().sum()),
            },
        }

    # 7. Check if a specific column is referenced
    column = _find_column(question, df.columns.tolist())

    if column is not None:
        series = df[column]

        # Missing in column
        if "missing" in normalized or "null" in normalized:
            miss_count = int(series.isna().sum())
            return {
                "answer": f"Column '{column}' has {miss_count:,} missing value(s).",
                "result": {"column": column, "missing_count": miss_count},
            }

        # Unique in column
        if "unique" in normalized or "distinct" in normalized:
            uniq_count = int(series.nunique(dropna=True))
            return {
                "answer": f"Column '{column}' has {uniq_count:,} distinct value(s).",
                "result": {"column": column, "unique_count": uniq_count},
            }

        # Top values / frequent / distribution
        if any(w in normalized for w in ("frequent", "mode", "top", "distribution", "breakdown", "counts")):
            top_vals = _top_values(series, limit=10)
            formatted = ", ".join(f"'{item['value']}': {item['count']}" for item in top_vals[:5])
            return {
                "answer": f"The most frequent values in '{column}' are: {formatted}.",
                "result": {"column": column, "top_values": top_vals},
            }

        # Numeric statistics
        if pd.api.types.is_numeric_dtype(series):
            operations = {
                "average": ("mean", series.mean),
                "mean": ("mean", series.mean),
                "median": ("median", series.median),
                "sum": ("sum", series.sum),
                "minimum": ("minimum", series.min),
                "min": ("minimum", series.min),
                "lowest": ("minimum", series.min),
                "maximum": ("maximum", series.max),
                "max": ("maximum", series.max),
                "highest": ("maximum", series.max),
                "standard deviation": ("standard deviation", series.std),
            }

            for phrase, (label, calculation) in operations.items():
                if phrase in normalized:
                    val = calculation()
                    if pd.isna(val):
                        return {
                            "answer": f"Column '{column}' has no valid numeric values to calculate {label}.",
                            "result": {"column": column, "metric": label, "value": None},
                        }
                    res_val = round(float(val), 4)
                    return {
                        "answer": f"The {label} of '{column}' is {res_val}.",
                        "result": {"column": column, "metric": label, "value": res_val},
                    }

            if "describe" in normalized or "statistic" in normalized:
                stats = _numeric_summary(series)
                return {
                    "answer": f"Descriptive statistics for '{column}': Mean = {stats.get('mean')}, Min = {stats.get('min')}, Max = {stats.get('max')}.",
                    "result": {"column": column, "statistics": stats},
                }

        # Count of non-missing in column
        if "count" in normalized or "how many" in normalized:
            cnt = int(series.notna().sum())
            return {
                "answer": f"Column '{column}' has {cnt:,} non-missing value(s).",
                "result": {"column": column, "non_missing_count": cnt},
            }

        # If user asks about a column without specific operation, provide quick profile
        if pd.api.types.is_numeric_dtype(series):
            return {
                "answer": f"Column '{column}' is numeric (min: {series.min()}, mean: {round(float(series.mean()), 2)}, max: {series.max()}).",
                "result": {"column": column, "summary": _numeric_summary(series)},
            }
        else:
            top_vals = _top_values(series, limit=5)
            formatted = ", ".join(f"'{item['value']}': {item['count']}" for item in top_vals)
            return {
                "answer": f"Column '{column}' is categorical with {series.nunique():,} unique values. Most common: {formatted}.",
                "result": {"column": column, "top_values": top_vals},
            }

    # 8. Try Groq LLM if available online
    llm_answer = _ask_groq_llm(df, question, chat_history)
    if llm_answer is not None:
        return llm_answer

    # 9. Fallback response
    return {
        "answer": (
            f"I could not match that question to a specific column or operation. "
            f"The dataset has {len(df):,} rows and {len(df.columns)} columns: {', '.join(df.columns)}. "
            "You can ask about duplicates, missing values, column counts, unique values, averages, or distributions."
        ),
        "result": {
            "columns": df.columns.tolist(),
            "rows": len(df),
            "duplicates": int(df.duplicated().sum()),
        },
    }
