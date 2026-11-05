"""Dataset comparison reporter module.

Compares an original DataFrame and a cleaned DataFrame across key data-quality
metrics (duplicates, missing values, unique values, numeric/categorical types)
using structured tables and visual charts. Works dynamically with any dataset.
"""

import os
import tempfile
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Prevent matplotlib cache permission warning on Windows
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

# Use non-interactive backend for headless/terminal environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_column_classification(df: pd.DataFrame) -> dict:
    """Classify columns dynamically into numeric and categorical (non-numeric)."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
    }


def compute_comparison_metrics(df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """Compute comprehensive comparison metrics between original and cleaned DataFrames."""
    orig_class = _get_column_classification(df)
    clean_class = _get_column_classification(cleaned_df)

    orig_dup = int(df.duplicated().sum())
    clean_dup = int(cleaned_df.duplicated().sum())

    orig_total_missing = int(df.isna().sum().sum())
    clean_total_missing = int(cleaned_df.isna().sum().sum())

    # Align columns dynamically (handles columns added/removed/renamed)
    all_columns = list(dict.fromkeys(list(df.columns) + list(cleaned_df.columns)))

    column_metrics = []
    for col in all_columns:
        in_orig = col in df.columns
        in_clean = col in cleaned_df.columns

        orig_dtype = str(df[col].dtype) if in_orig else "N/A"
        clean_dtype = str(cleaned_df[col].dtype) if in_clean else "N/A"

        orig_missing = int(df[col].isna().sum()) if in_orig else 0
        clean_missing = int(cleaned_df[col].isna().sum()) if in_clean else 0
        missing_delta = clean_missing - orig_missing

        orig_unique = int(df[col].nunique(dropna=True)) if in_orig else 0
        clean_unique = int(cleaned_df[col].nunique(dropna=True)) if in_clean else 0
        unique_delta = clean_unique - orig_unique

        column_metrics.append({
            "column": col,
            "orig_dtype": orig_dtype,
            "clean_dtype": clean_dtype,
            "orig_missing": orig_missing,
            "clean_missing": clean_missing,
            "missing_delta": missing_delta,
            "orig_unique": orig_unique,
            "clean_unique": clean_unique,
            "unique_delta": unique_delta,
            "in_original": in_orig,
            "in_cleaned": in_clean,
        })

    return {
        "overview": {
            "orig_rows": len(df),
            "clean_rows": len(cleaned_df),
            "row_delta": len(cleaned_df) - len(df),
            "orig_cols": len(df.columns),
            "clean_cols": len(cleaned_df.columns),
            "orig_duplicates": orig_dup,
            "clean_duplicates": clean_dup,
            "duplicate_delta": clean_dup - orig_dup,
            "orig_total_missing": orig_total_missing,
            "clean_total_missing": clean_total_missing,
            "missing_delta": clean_total_missing - orig_total_missing,
        },
        "column_classification": {
            "original": orig_class,
            "cleaned": clean_class,
        },
        "columns": column_metrics,
    }


def _ascii_bar(val: int, max_val: int, length: int = 12) -> str:
    """Render a clean, CP1252/ASCII-compatible progress bar."""
    if max_val <= 0 or val <= 0:
        return "." * length
    filled = int(round((val / max_val) * length))
    filled = max(0, min(length, filled))
    return "#" * filled + "." * (length - filled)


def generate_ascii_visualizations(metrics: dict) -> str:
    """Create lightweight ASCII visual charts for terminal display."""
    lines = []
    lines.append("ASCII Visual Summary:")
    lines.append("-" * 65)

    # Missing values per column that had missing values
    col_missing = [
        c for c in metrics["columns"]
        if c["orig_missing"] > 0 or c["clean_missing"] > 0
    ]

    if col_missing:
        lines.append("Missing Values by Column (Before -> After):")
        max_missing = max(
            [c["orig_missing"] for c in col_missing]
            + [c["clean_missing"] for c in col_missing]
            + [1]
        )
        for c in col_missing:
            b_bar = _ascii_bar(c["orig_missing"], max_missing, 10)
            a_bar = _ascii_bar(c["clean_missing"], max_missing, 10)
            lines.append(
                f"  {c['column'][:25]:<25} | Before: [{b_bar}] {c['orig_missing']:>4} "
                f"-> After: [{a_bar}] {c['clean_missing']:>4} ({c['missing_delta']:+d})"
            )
    else:
        lines.append("Missing Values: No missing values found in original or cleaned dataset.")

    # Duplicate rows visual
    dup_orig = metrics["overview"]["orig_duplicates"]
    dup_clean = metrics["overview"]["clean_duplicates"]
    max_dup = max(dup_orig, dup_clean, 1)
    lines.append("\nDuplicate Rows:")
    lines.append(
        f"  Original: [{_ascii_bar(dup_orig, max_dup, 14)}] {dup_orig} row(s)\n"
        f"  Cleaned:  [{_ascii_bar(dup_clean, max_dup, 14)}] {dup_clean} row(s)"
    )

    # Column types breakdown
    orig_class = metrics["column_classification"]["original"]
    clean_class = metrics["column_classification"]["cleaned"]
    tot_cols = max(metrics["overview"]["orig_cols"], metrics["overview"]["clean_cols"], 1)

    lines.append("\nColumn Types Breakdown:")
    lines.append(
        f"  Original: Numeric: [{_ascii_bar(orig_class['numeric_count'], tot_cols, 10)}] {orig_class['numeric_count']:>2} | "
        f"Categorical: [{_ascii_bar(orig_class['categorical_count'], tot_cols, 10)}] {orig_class['categorical_count']:>2}"
    )
    lines.append(
        f"  Cleaned:  Numeric: [{_ascii_bar(clean_class['numeric_count'], tot_cols, 10)}] {clean_class['numeric_count']:>2} | "
        f"Categorical: [{_ascii_bar(clean_class['categorical_count'], tot_cols, 10)}] {clean_class['categorical_count']:>2}"
    )
    lines.append("-" * 65)

    return "\n".join(lines)


def save_comparison_chart(
    df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    output_path: str = "reports/dataset_comparison_report.png",
) -> str | None:
    """Generate a multi-panel visual comparison chart and save to file."""
    try:
        metrics = compute_comparison_metrics(df, cleaned_df)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Dataset Quality Comparison: Original vs Cleaned", fontsize=15, fontweight="bold")

        # 1. Duplicates and Total Missing
        ax1 = axes[0, 0]
        categories = ["Duplicate Rows", "Total Missing Values"]
        before_vals = [metrics["overview"]["orig_duplicates"], metrics["overview"]["orig_total_missing"]]
        after_vals = [metrics["overview"]["clean_duplicates"], metrics["overview"]["clean_total_missing"]]

        x = range(len(categories))
        width = 0.35
        ax1.bar([i - width/2 for i in x], before_vals, width, label="Original (df)", color="#e74c3c", alpha=0.85)
        ax1.bar([i + width/2 for i in x], after_vals, width, label="Cleaned (cleaned_df)", color="#2ecc71", alpha=0.85)
        ax1.set_xticks(list(x))
        ax1.set_xticklabels(categories, fontweight="semibold")
        ax1.set_ylabel("Count")
        ax1.set_title("Overall Data Quality Issues")
        ax1.legend()
        ax1.grid(axis="y", linestyle="--", alpha=0.6)

        # 2. Column Types (Numeric vs Categorical)
        ax2 = axes[0, 1]
        type_cats = ["Numeric Columns", "Categorical Columns"]
        orig_types = [
            metrics["column_classification"]["original"]["numeric_count"],
            metrics["column_classification"]["original"]["categorical_count"],
        ]
        clean_types = [
            metrics["column_classification"]["cleaned"]["numeric_count"],
            metrics["column_classification"]["cleaned"]["categorical_count"],
        ]
        x2 = range(len(type_cats))
        ax2.bar([i - width/2 for i in x2], orig_types, width, label="Original (df)", color="#3498db", alpha=0.85)
        ax2.bar([i + width/2 for i in x2], clean_types, width, label="Cleaned (cleaned_df)", color="#9b59b6", alpha=0.85)
        ax2.set_xticks(list(x2))
        ax2.set_xticklabels(type_cats, fontweight="semibold")
        ax2.set_ylabel("Number of Columns")
        ax2.set_title("Column Type Distribution")
        ax2.legend()
        ax2.grid(axis="y", linestyle="--", alpha=0.6)

        # 3. Missing values per column (only columns with missing values)
        ax3 = axes[1, 0]
        missing_cols = [
            c for c in metrics["columns"]
            if c["orig_missing"] > 0 or c["clean_missing"] > 0
        ]
        if missing_cols:
            col_names = [c["column"] for c in missing_cols][:10]  # top 10
            orig_m = [c["orig_missing"] for c in missing_cols][:10]
            clean_m = [c["clean_missing"] for c in missing_cols][:10]
            y = range(len(col_names))
            h = 0.35
            ax3.barh([i - h/2 for i in y], orig_m, h, label="Original", color="#e74c3c", alpha=0.85)
            ax3.barh([i + h/2 for i in y], clean_m, h, label="Cleaned", color="#2ecc71", alpha=0.85)
            ax3.set_yticks(list(y))
            ax3.set_yticklabels(col_names, fontsize=8)
            ax3.set_xlabel("Missing Values Count")
            ax3.set_title("Missing Values by Column (Top Columns)")
            ax3.legend()
            ax3.grid(axis="x", linestyle="--", alpha=0.6)
        else:
            ax3.text(0.5, 0.5, "No Missing Values Found", ha="center", va="center", fontsize=12)
            ax3.set_title("Missing Values by Column")

        # 4. Unique values per column
        ax4 = axes[1, 1]
        top_cols = metrics["columns"][:10]  # first 10 columns
        col_names = [c["column"] for c in top_cols]
        orig_u = [c["orig_unique"] for c in top_cols]
        clean_u = [c["clean_unique"] for c in top_cols]
        y4 = range(len(col_names))
        h = 0.35
        ax4.barh([i - h/2 for i in y4], orig_u, h, label="Original", color="#3498db", alpha=0.85)
        ax4.barh([i + h/2 for i in y4], clean_u, h, label="Cleaned", color="#1abc9c", alpha=0.85)
        ax4.set_yticks(list(y4))
        ax4.set_yticklabels(col_names, fontsize=8)
        ax4.set_xlabel("Unique Values Count")
        ax4.set_title("Unique Values by Column (First 10 Columns)")
        ax4.legend()
        ax4.grid(axis="x", linestyle="--", alpha=0.6)

        plt.tight_layout()
        plt.savefig(out_file, dpi=150)
        plt.close(fig)
        return str(out_file.resolve())
    except Exception as exc:
        print(f"[Warning] Could not generate comparison chart: {exc}")
        return None


def display_dataset_report(
    df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    chart_output_path: str = "reports/dataset_comparison_report.png",
) -> dict:
    """Print a comprehensive comparison report to the console and generate a visual chart.

    Displays:
    1. Overall summary comparison table (rows, duplicates, total missing, types)
    2. Column classification breakdown (numeric vs categorical column names)
    3. Column-by-column detailed table (dtype, missing count, unique count)
    4. ASCII visualization in terminal
    5. Path to the saved Matplotlib visual comparison figure
    """
    metrics = compute_comparison_metrics(df, cleaned_df)
    overview = metrics["overview"]
    orig_class = metrics["column_classification"]["original"]
    clean_class = metrics["column_classification"]["cleaned"]

    print("\n" + "=" * 80)
    print("                      DATASET COMPARISON REPORT                       ")
    print("                 Original (df) vs. Cleaned (cleaned_df)               ")
    print("=" * 80)

    # 1. High-level Overview Table
    summary_data = [
        ["Total Rows", overview["orig_rows"], overview["clean_rows"], f"{overview['row_delta']:+d}"],
        ["Total Columns", overview["orig_cols"], overview["clean_cols"], f"{overview['clean_cols'] - overview['orig_cols']:+d}"],
        ["Duplicate Rows", overview["orig_duplicates"], overview["clean_duplicates"], f"{overview['duplicate_delta']:+d}"],
        ["Total Missing Values", overview["orig_total_missing"], overview["clean_total_missing"], f"{overview['missing_delta']:+d}"],
        ["Numeric Columns Count", orig_class["numeric_count"], clean_class["numeric_count"], f"{clean_class['numeric_count'] - orig_class['numeric_count']:+d}"],
        ["Categorical Columns Count", orig_class["categorical_count"], clean_class["categorical_count"], f"{clean_class['categorical_count'] - orig_class['categorical_count']:+d}"],
    ]
    print("\n### 1. Overall Dataset Metrics Summary")
    print(tabulate(
        summary_data,
        headers=["Metric", "Original (df)", "Cleaned (cleaned_df)", "Change"],
        tablefmt="github",
    ))

    # 2. Column Classification Table & Names
    print("\n### 2. Numeric and Categorical Column Breakdown")
    type_data = [
        [
            "Numeric",
            orig_class["numeric_count"],
            ", ".join(orig_class["numeric_cols"]) if orig_class["numeric_cols"] else "(none)",
            clean_class["numeric_count"],
            ", ".join(clean_class["numeric_cols"]) if clean_class["numeric_cols"] else "(none)",
        ],
        [
            "Categorical",
            orig_class["categorical_count"],
            ", ".join(orig_class["categorical_cols"]) if orig_class["categorical_cols"] else "(none)",
            clean_class["categorical_count"],
            ", ".join(clean_class["categorical_cols"]) if clean_class["categorical_cols"] else "(none)",
        ],
    ]
    print(tabulate(
        type_data,
        headers=["Type", "Orig Count", "Original Column Names", "Clean Count", "Cleaned Column Names"],
        tablefmt="github",
    ))

    # 3. Column-by-Column Detailed Comparison Table
    print("\n### 3. Column-Level Metrics (Missing Values & Unique Counts)")
    col_rows = []
    for c in metrics["columns"]:
        changes = []
        if c["orig_dtype"] != c["clean_dtype"]:
            changes.append(f"dtype: {c['orig_dtype']}->{c['clean_dtype']}")
        if c["missing_delta"] != 0:
            changes.append(f"missing: {c['orig_missing']}->{c['clean_missing']}")
        if c["unique_delta"] != 0:
            changes.append(f"unique: {c['orig_unique']}->{c['clean_unique']}")

        status_str = ", ".join(changes) if changes else "Unchanged"

        col_rows.append([
            c["column"],
            f"{c['orig_dtype']} -> {c['clean_dtype']}",
            f"{c['orig_missing']} -> {c['clean_missing']} ({c['missing_delta']:+d})",
            f"{c['orig_unique']} -> {c['clean_unique']} ({c['unique_delta']:+d})",
            status_str,
        ])

    print(tabulate(
        col_rows,
        headers=[
            "Column Name",
            "Dtype (Orig -> Clean)",
            "Missing Values (Orig -> Clean)",
            "Unique Values (Orig -> Clean)",
            "Changes Observed",
        ],
        tablefmt="github",
    ))

    # 4. ASCII Visual Charts
    print(f"\n### 4. Visualizations")
    ascii_summary = generate_ascii_visualizations(metrics)
    print(ascii_summary)

    # 5. Graphical Visual Comparison Chart
    saved_chart = save_comparison_chart(df, cleaned_df, chart_output_path)
    if saved_chart:
        print(f"\nVisual comparison chart generated and saved to: {saved_chart}")

    print("=" * 80 + "\n")
    return metrics
