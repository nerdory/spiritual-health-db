"""Assessment configuration loaded from Streamlit secrets.

`secrets.toml` (or the Streamlit Cloud secrets editor) holds the assessment in
a per-question layout that mirrors the master question doc: under
`[assessment.questions.<track>.<axis>]`, each key is a numbered question and
its value is the list of answer options in ASCENDING score order — the first
option scores 1 point, the last scores 5. `axis_mapping` defines which question
numbers make up each chart axis (and their display order); `axis_colors` is an
optional per-axis color override.

Matching a submitted answer to a score is tolerant of minor doc-vs-form drift:
exact match first, then a unique-prefix match, otherwise None (excluded).
"""

import streamlit as st

_DEFAULT_AXIS_COLOR = "#1F77B4"


def _assessment() -> dict:
    """Returns the [assessment] section from secrets.

    Raises:
        ValueError: If the section is missing.
    """
    config = st.secrets.get("assessment")
    if not config:
        raise ValueError(
            "`assessment` configuration is missing from secrets. Add "
            "[assessment.axis_mapping], [assessment.questions.<track>.<axis>], and "
            "optionally [assessment.axis_colors] to `.streamlit/secrets.toml` or the "
            "Streamlit Cloud secrets editor."
        )
    return config


def _questions() -> dict:
    """Returns the [assessment.questions] section (tracks -> axes -> questions)."""
    return _assessment().get("questions", {})


def _question_num(question_key: str) -> int:
    """Extracts the leading question number from a '12. Question text' key."""
    return int(question_key.split(".", 1)[0])


def get_axis_mapping() -> dict[str, list[int]]:
    """Returns ordered canonical axis name -> question numbers.

    Raises:
        ValueError: If the mapping is missing.
    """
    mapping = _assessment().get("axis_mapping")
    if not mapping:
        raise ValueError("`assessment.axis_mapping` is missing from secrets.")
    return {str(axis): [int(q) for q in q_nums] for axis, q_nums in mapping.items()}


def get_axis_order() -> list[str]:
    """Returns canonical axis names in `axis_mapping` order."""
    return list(get_axis_mapping().keys())


def get_axis_colors() -> dict[str, str]:
    """Returns canonical axis name -> hex color, defaulting per missing axis."""
    colors = _assessment().get("axis_colors", {})
    return {axis: str(colors.get(axis, _DEFAULT_AXIS_COLOR)) for axis in get_axis_order()}


def _build_text_to_score() -> dict[str, int]:
    """Builds answer option text -> score from option positions across all tracks."""
    mapping: dict[str, int] = {}
    for axes in _questions().values():
        for questions in axes.values():
            for options in questions.values():
                for idx, option in enumerate(options):
                    mapping[str(option)] = idx + 1
    return mapping


def get_text_to_score() -> dict[str, int]:
    """Returns every configured answer option -> its 1..5 score."""
    return _build_text_to_score()


def get_total_questions() -> int:
    """Returns the number of scored questions, derived from the axis mapping."""
    return max(q for q_nums in get_axis_mapping().values() for q in q_nums)


def score_answer(answer: str) -> int | None:
    """Scores a submitted answer against the configured options.

    Exact match first; otherwise a unique-prefix match in either direction
    (the submitted answer may be a slightly shortened or lengthened form of
    the configured option).

    Args:
        answer (str): Raw answer text from the sheet.

    Returns:
        int | None: 1..5 score, or None if no unique match exists.
    """
    answer = str(answer).strip()
    text_to_score = _build_text_to_score()
    if answer in text_to_score:
        return text_to_score[answer]
    matches = {
        s for opt, s in text_to_score.items() if opt.startswith(answer) or answer.startswith(opt)
    }
    if len(matches) == 1:
        return next(iter(matches))
    return None
