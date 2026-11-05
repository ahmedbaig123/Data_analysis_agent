import pandas as pd


def _normalise_numeric_values(values: pd.Series) -> pd.Series:
    """Remove common display formatting before numeric conversion."""
    values = values.astype("string").str.strip()
    values = values.str.replace(r"[$€£₹¥,]", "", regex=True)
    return values.str.replace(
        r"(?i)\s*(?:hrs?|hours?|yrs?|years?)\s*$", "", regex=True
    )


def find_numeric_like_columns(df: pd.DataFrame) -> dict:
    candidates = {}

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            continue

        values = _normalise_numeric_values(df[column].dropna())

        if values.empty:
            continue

        numeric_values = pd.to_numeric(values, errors="coerce")
        ratio = numeric_values.notna().mean()

        if ratio >= 0.8:
            candidates[column] = round(float(ratio), 2)

    return candidates


def clean_dataset(df: pd.DataFrame, actions: list[dict]):
    df = df.copy()
    log = []

    for item in actions:
        action = item["action"]
        column = item.get("column")

        if action == "remove_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            log.append({
                "action": action,
                "rows_affected": before - len(df),
            })

        elif action == "trim_whitespace":
            columns = (
                df.select_dtypes(include=["object", "string"]).columns
                if column == "all_string_columns"
                else [column]
            )
            affected_count = 0

            for target_column in columns:
                before = df[target_column].copy()
                df[target_column] = df[target_column].apply(
                    lambda value: value.strip()
                    if isinstance(value, str)
                    else value
                )
                affected = before.ne(df[target_column]) & before.notna()
                affected_count += int(affected.sum())

            log.append({
                "action": action,
                "column": column,
                "values_affected": affected_count,
            })

        elif action == "convert_to_numeric":
            non_missing_before = df[column].notna()
            converted = pd.to_numeric(
                _normalise_numeric_values(df[column]),
                errors="coerce",
            )
            df[column] = converted

            log.append({
                "action": action,
                "column": column,
                "values_affected": int(converted.notna().sum()),
                "values_became_missing": int(
                    (non_missing_before & converted.isna()).sum()
                ),
            })

        elif action == "standardize_case":
            before = df[column].copy()
            df[column] = df[column].apply(
                lambda value: value.title()
                if isinstance(value, str)
                else value
            )
            affected = before.ne(df[column]) & before.notna()

            log.append({
                "action": action,
                "column": column,
                "values_affected": int(affected.sum()),
            })

        elif action == "fill_missing":
            missing = df[column].isna()
            strategy = item["strategy"]
            fill_value = None
            skip_reason = None

            if strategy in {"median", "mean"}:
                numeric_values = pd.to_numeric(
                    _normalise_numeric_values(df[column]),
                    errors="coerce",
                )

                # Do not fill if an existing value is not numeric.
                has_non_numeric = (
                    df[column].notna() & numeric_values.isna()
                ).any()

                if has_non_numeric:
                    skip_reason = (
                        "Column contains non-numeric values; "
                        "left missing values unchanged."
                    )
                else:
                    fill_value = (
                        numeric_values.median()
                        if strategy == "median"
                        else numeric_values.mean()
                    )

                    # Convert numeric-looking text to numeric values.
                    df[column] = numeric_values

            elif strategy == "mode":
                modes = df.loc[~missing, column].mode()

                if not modes.empty:
                    fill_value = modes.iloc[0]

            if (
                strategy != "leave_missing"
                and skip_reason is None
                and pd.notna(fill_value)
            ):
                if strategy in {"mean", "median"}:
                    df[column] = df[column].astype("Float64")

                df.loc[missing, column] = fill_value

            log.append({
                "action": action,
                "column": column,
                "strategy": strategy,
                "values_affected": (
                    int(missing.sum())
                    if skip_reason is None and pd.notna(fill_value)
                    else 0
                ),
                "skipped": skip_reason,
            })

        elif action == "flag_for_review":
            log.append({
                "action": action,
                "column": column,
                "reason": item.get("reason", ""),
                "values_affected": 0,
            })

    return df, log