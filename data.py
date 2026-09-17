"""Data processing module for Spiritual Health Assessment.

Handles fetching data from Google Sheets, mapping response text to Likert scale values,
deduplicating entries to retain only the latest submissions for group/axis metrics, and
preserving the full per-submission history for longitudinal progress tracking.
"""

import re
import pandas as pd
import streamlit as st

from assessment_config import get_axis_mapping, get_total_questions, score_answer

# Assessment axes and per-question answer scoring come from secrets (see
# assessment_config.py); edit them there, not in code.
AXIS_MAPPING: dict[str, list[int]] = get_axis_mapping()
TOTAL_QUESTIONS: int = get_total_questions()


def generate_member_id(name: str, pin: str) -> str:
    """Generates a standardized, deterministic Member ID from name and PIN.

    Args:
        name (str): Participant's preferred name.
        pin (str): Participant's chosen 4-digit PIN.

    Returns:
        str: Alphanumeric Member ID (e.g., 'John1234').
    """
    clean_name = re.sub(r"[^a-zA-Z0-9]", "", str(name)).title()
    clean_pin = re.sub(r"[^0-9]", "", str(pin)).zfill(4)
    return f"{clean_name}{clean_pin}"


def parse_consent(answer) -> bool:
    """Interprets a consent answer as a boolean.

    Matches "Consent" in the question title (see `load_data`) and inspects the
    answer text: an explicit "Yes" grants consent, "No" withholds it. Anything
    else (blank, unrecognized wording) is treated as NO consent, so a missing
    or ambiguous answer never exposes a member's results by accident.

    Args:
        answer: Raw consent answer text from the sheet (may be NaN).

    Returns:
        bool: True only when the member explicitly agreed.
    """
    if pd.isna(answer):
        return False
    text = str(answer).strip()
    if re.search(r"\byes\b", text, re.IGNORECASE):
        return True
    if re.search(r"\bno\b", text, re.IGNORECASE):
        return False
    return False


def load_google_sheet_data() -> pd.DataFrame:
    """Fetches response records directly from the public Google Sheet export URL.

    Returns:
        pd.DataFrame: Raw response data from the Google Sheet.

    Raises:
        ValueError: If sheet_id is not set in Streamlit secrets.
    """
    sheet_id = st.secrets.get("sheet_id")
    if not sheet_id:
        raise ValueError("`sheet_id` is not defined in `.streamlit/secrets.toml`.")

    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    df = pd.read_csv(csv_url)
    df.columns = df.columns.str.strip()
    return df


def _map_scores(df: pd.DataFrame, all_q_cols: list[str]) -> pd.DataFrame:
    """Maps raw answer text to Likert numbers (exact or unique-prefix match).

    Args:
        df (pd.DataFrame): Rows carrying raw answer text in `all_q_cols`.
        all_q_cols (list[str]): Sheet question column names, in order.

    Returns:
        pd.DataFrame: Copy with question columns replaced by numeric scores.
    """
    mapped_df = df.copy()
    for col in all_q_cols:
        mapped_df[col] = mapped_df[col].map(lambda v: score_answer(v) if pd.notna(v) else None)
        mapped_df[col] = pd.to_numeric(mapped_df[col], errors="coerce")
    return mapped_df


def _question_frames(
    mapped_df: pd.DataFrame, all_q_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Builds wide Q1..Qn score columns plus the long question-level frame.

    Args:
        mapped_df (pd.DataFrame): Scored submissions; must carry `person_id`,
            `take`, and `clean_timestamp`.
        all_q_cols (list[str]): Sheet question column names, in order.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - mapped_df merged with Q1..Qn wide columns and per-axis means.
            - Long frame, one row per (submission, question): person_id, take,
              timestamp, q_num, q_int, score, axis, question.
    """
    q_rows = []
    q_long_rows = []

    for _, row in mapped_df.iterrows():
        non_null_scores = row[all_q_cols].dropna()
        q_scores = non_null_scores.values.tolist()[:TOTAL_QUESTIONS]
        q_headers = non_null_scores.index.tolist()[:TOTAL_QUESTIONS]
        q_dict = {f"Q{i+1}": (q_scores[i] if i < len(q_scores) else None) for i in range(TOTAL_QUESTIONS)}
        q_dict["person_id"] = row["person_id"]
        q_dict["take"] = row["take"]
        q_rows.append(q_dict)

        for i, score in enumerate(q_scores):
            q_num = i + 1
            axis_name = next((axis for axis, q_list in AXIS_MAPPING.items() if q_num in q_list), "Unknown")
            question_text = re.sub(r"\s+", " ", str(q_headers[i])).strip()
            # Drop the sheet header's own numbering ("13. ...") so "Q13" stays the single label
            question_text = re.sub(rf"^{q_num}\s*[.)]\s+", "", question_text)
            q_long_rows.append({
                "person_id": row["person_id"],
                "take": row["take"],
                "timestamp": row["clean_timestamp"],
                "q_num": f"Q{q_num}",
                "q_int": q_num,
                "score": score,
                "axis": axis_name,
                "question": question_text,
            })

    q_df = pd.DataFrame(q_rows)
    df = mapped_df.merge(q_df, on=["person_id", "take"], how="left")

    for axis_name, q_indices in AXIS_MAPPING.items():
        cols = [f"Q{i}" for i in q_indices]
        df[axis_name] = df[cols].mean(axis=1)

    return df, pd.DataFrame(q_long_rows)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches, parses, deduplicates, and aggregates Google Sheet data.

    Group/axis metrics deduplicate by retaining only each member's latest
    submission, while the full submission history is preserved for
    longitudinal tracking (past submissions keep the group/track context
    they were submitted under).

    Members who decline the optional leader-sharing consent question are flagged
    with `has_consent = False` so the Leader Dashboard can mask their identity,
    scores, and charts.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - people: Clean DataFrame of unique members (latest submission).
            - responses: Aggregated 6-axis category scores per member (latest only).
            - q_responses: Question-level responses (Q1-Q24) per member (latest only).
            - submissions: One row per submission ever made — person_id, name,
              group, is_believer, has_consent, timestamp, take (1 = first), and
              axis means.
            - q_history: Question-level responses for every submission, with
              take number and timestamp, chronological per member.
    """
    raw_df = load_google_sheet_data()
    if raw_df.empty:
        return (pd.DataFrame(),) * 5

    df = raw_df.copy()

    name_col = next((c for c in df.columns if "First/Preferred Name" in c or "Name" in c), None)
    pin_col = next((c for c in df.columns if "PIN" in c), None)
    group_col = next((c for c in df.columns if "group" in c.lower()), None)
    journey_col = next((c for c in df.columns if "spiritual journey" in c.lower()), None)
    # Optional leader-sharing consent question, matched by "Consent" in the title.
    consent_col = next((c for c in df.columns if "consent" in c.lower()), None)

    if not all([name_col, pin_col, group_col]):
        missing = [c for c, v in [("Name", name_col), ("PIN", pin_col), ("Group", group_col)] if v is None]
        raise ValueError(f"Missing required sheet column(s): {', '.join(missing)}")

    df["clean_name"] = df[name_col].astype(str).str.strip()
    df["clean_pin"] = df[pin_col].astype(str).str.strip()
    df["clean_group"] = df[group_col].astype(str).str.strip()
    df["person_id"] = df.apply(lambda r: generate_member_id(r["clean_name"], r["clean_pin"]), axis=1)

    if journey_col:
        df["is_believer"] = df[journey_col].astype(str).str.contains("follower of Jesus", case=False)
    else:
        df["is_believer"] = True

    # Per-submission consent flag. A sheet with no consent question keeps the
    # previous behavior (everything visible to leaders).
    if consent_col:
        df["has_consent"] = df[consent_col].map(parse_consent)
    else:
        df["has_consent"] = True

    timestamp_col = "Timestamp" if "Timestamp" in df.columns else df.columns[0]
    df["clean_timestamp"] = pd.to_datetime(df[timestamp_col], errors="coerce")

    # Chronological order per member; take 1 = their first submission.
    df = df.sort_values("clean_timestamp")
    df["take"] = df.groupby("person_id").cumcount() + 1

    metadata_cols = {
        name_col, pin_col, group_col, journey_col, consent_col, timestamp_col,
        "clean_name", "clean_pin", "clean_group", "person_id", "is_believer",
        "has_consent", "clean_timestamp", "take",
    }
    all_q_cols = [c for c in df.columns if c not in metadata_cols]

    # Latest-only frames (group averages must never double-count a member):
    # dedup BEFORE score mapping, exactly the historical behavior.
    latest_df = df.groupby("person_id", as_index=False).last()
    latest_merged, q_responses = _question_frames(_map_scores(latest_df, all_q_cols), all_q_cols)

    people = latest_merged[
        ["person_id", "clean_name", "clean_group", "is_believer", "has_consent", "clean_timestamp"]
    ].copy()
    people.rename(columns={"clean_name": "name", "clean_group": "group", "clean_timestamp": "timestamp"}, inplace=True)

    response_rows = []
    for _, row in latest_merged.iterrows():
        for category in AXIS_MAPPING.keys():
            response_rows.append({
                "person_id": row["person_id"],
                "name": row["clean_name"],
                "group": row["clean_group"],
                "is_believer": row["is_believer"],
                "timestamp": row["clean_timestamp"],
                "category": category,
                "score": row[category],
            })

    responses = pd.DataFrame(response_rows)

    # Full-history frames: every submission kept, scored in its own context.
    history_merged, q_history = _question_frames(_map_scores(df, all_q_cols), all_q_cols)

    submissions = history_merged[
        ["person_id", "clean_name", "clean_group", "is_believer", "has_consent",
         "clean_timestamp", "take", *AXIS_MAPPING.keys()]
    ].copy()
    submissions.rename(
        columns={"clean_name": "name", "clean_group": "group", "clean_timestamp": "timestamp"},
        inplace=True,
    )
    submissions = submissions.sort_values(["person_id", "take"]).reset_index(drop=True)
    q_history = q_history.sort_values(["person_id", "take", "q_int"]).reset_index(drop=True)

    return people, responses, q_responses, submissions, q_history


def person_history(
    submissions_df: pd.DataFrame, q_history_df: pd.DataFrame, person_id: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches all historical responses for one member in chronological order.

    Past submissions are never dropped, and each row keeps the group/track
    context it was originally submitted under (e.g. a member who transitioned
    from Exploring Faith to Follower of Jesus shows their old takes as-is).

    Args:
        submissions_df (pd.DataFrame): Full submission-level history frame.
        q_history_df (pd.DataFrame): Full question-level history frame.
        person_id (str): Member ID to look up.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - submissions: One row per take, chronological (take 1 first).
            - q_history: Question-level rows for every take.
    """
    subs = (
        submissions_df[submissions_df["person_id"] == person_id]
        .sort_values("take")
        .reset_index(drop=True)
    )
    q_hist = q_history_df[q_history_df["person_id"] == person_id].copy()
    return subs, q_hist


def submission_category_scores(submission_row: pd.Series) -> pd.DataFrame:
    """Reshapes one submission row into the category/score frame charts expect.

    Args:
        submission_row (pd.Series): A single row from the submissions frame.

    Returns:
        pd.DataFrame: One row per axis with `category` and `score` columns.
    """
    return pd.DataFrame({
        "person_id": submission_row["person_id"],
        "name": submission_row["name"],
        "category": list(AXIS_MAPPING.keys()),
        "score": [submission_row[axis] for axis in AXIS_MAPPING.keys()],
    })


def category_means(responses_df: pd.DataFrame) -> pd.Series:
    """Computes deduplicated average scores for each category axis.

    Args:
        responses_df (pd.DataFrame): Category-level response scores.

    Returns:
        pd.Series: Mean scores indexed by category.
    """
    if responses_df.empty:
        return pd.Series(dtype=float)
    return responses_df.groupby("category")["score"].mean()


def group_responses(people_df: pd.DataFrame, responses_df: pd.DataFrame, group_name: str) -> pd.DataFrame:
    """Filters responses for unique members belonging to a specific group.

    Args:
        people_df (pd.DataFrame): DataFrame of unique group members.
        responses_df (pd.DataFrame): Complete responses DataFrame.
        group_name (str): Target group name.

    Returns:
        pd.DataFrame: Filtered response rows for group members.
    """
    group_member_ids = people_df[people_df["group"] == group_name]["person_id"]
    return responses_df[responses_df["person_id"].isin(group_member_ids)].copy()
