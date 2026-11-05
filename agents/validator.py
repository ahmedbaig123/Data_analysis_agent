# Keep this list in sync with the action branches in cleaner.clean_dataset.
# Actions outside it are rejected before the cleaner can silently skip them.
IMPLEMENTED_ACTIONS = {
    "remove_duplicates",
    "convert_to_numeric",
    "standardize_case",
    "trim_whitespace",
    "fill_missing",
    "flag_for_review",
}

ALLOWED_MISSING_STRATEGIES = {
    "median",
    "mean",
    "mode",
    "leave_missing",
}

def validate_cleaning_plan(plan: dict,columns: list[str]) -> dict:
    """Return valid actions and rejected actions without changing the dataset."""

    valid_actions = []
    rejected_actions=[]

    actions = plan.get("actions",[])

    if not isinstance(actions, list):
        return {
             "valid_actions": [],
             "rejected_actions": ["Plan must contain an actions list."],
        }

    for action in actions:
        if  not isinstance(action, dict):
            rejected_actions.append(f"Invalid action format: {action!r}")
            continue

        action_type = action.get("action")
        column = action.get("column")

        if action_type not in IMPLEMENTED_ACTIONS:
            rejected_actions.append(f"Unsupported action: {action_type!r}")
            continue

        if action_type == "remove_duplicates":
            valid_actions.append(action)
            continue

        if (
            column not in columns
            and not (
                action_type == "trim_whitespace"
                and column == "all_string_columns"
            )
        ):
            rejected_actions.append(
                f"Unknown column {column!r} for action {action_type!r}"
            )
            continue

        if action_type == "fill_missing":
            strategy = action.get("strategy")
            if strategy not in ALLOWED_MISSING_STRATEGIES:
                rejected_actions.append(
                    f"Invalid missing-value strategy: {strategy!r}"
                )
                continue

        valid_actions.append(action)

    return {
        "valid_actions": valid_actions,
        "rejected_actions": rejected_actions,
    } 



