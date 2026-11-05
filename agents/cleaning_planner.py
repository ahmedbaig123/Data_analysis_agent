import json
import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file.")

client = Groq(api_key=api_key)


def create_cleaning_plan(
    profile: dict,
    numeric_candidates: dict,
    duplicate_rows: int,
) -> dict:
    prompt = f"""
You are a careful data-cleaning planner.

Review the dataset profile, numeric-like column candidates, and exact
duplicate-row count. Propose useful cleaning actions for this dataset.

Allowed actions:
- remove_duplicates: remove exact duplicate rows.
- convert_to_numeric: convert a column to numeric when its values support it.
- standardize_case: normalize capitalization in a categorical text column.
- trim_whitespace: remove leading and trailing whitespace from text.
- fill_missing: fill missing values using a suitable strategy.
- flag_for_review: identify an issue that should not be changed automatically.

For fill_missing, use one of these strategies:
- median
- mean
- mode
- leave_missing

Rules:
- Do not convert ID or identifier columns to numbers.
- Do not invent column names, values, or counts.
- Do not fill missing IDs or free-text fields automatically. Use
  flag_for_review or leave_missing for these.
- Use median or mean only for numeric measurement columns where appropriate.
- Use mode only for categorical columns when it is reasonable.
- If the correct action is uncertain, use flag_for_review.
- Include a brief reason for every action.
- Return JSON only, matching this structure:

{{
  "actions": [
    {{
      "action": "fill_missing",
      "column": "column name",
      "strategy": "median",
      "reason": "brief reason"
    }}
  ]
}}

Dataset profile:
{json.dumps(profile, ensure_ascii=False)}

Numeric-like column candidates:
{json.dumps(numeric_candidates, ensure_ascii=False)}

Exact duplicate rows:
{duplicate_rows}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "Return valid JSON only. Do not include Markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)