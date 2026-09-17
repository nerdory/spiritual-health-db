"""Main Streamlit application entrypoint for Spiritual Health Assessment.

Provides Individual View and Group Leader View with simplified controls, plus
longitudinal progress tracking across repeated submissions.
Run with: streamlit run app.py
"""

import html

import pandas as pd
import streamlit as st

from charts import (
    AXIS_COLORS,
    AXIS_ORDER,
    clean_axis_label,
    get_dynamic_color_and_emoji,
    group_radar_chart,
    individual_question_dot_plot,
    polar_scatter_chart,
    question_trend_sparkline,
)
from data import (
    category_means,
    group_responses,
    load_data,
    person_history,
    submission_category_scores,
)

st.set_page_config(page_title="Spiritual Health Assessment", page_icon="🕊️", layout="wide")

# Render-level interaction lock: no scroll/double-click zoom, no modebar
# (which carries the zoom buttons). Hover tooltips stay enabled.
PLOTLY_CONFIG: dict = {
    "displayModeBar": False,
    "scrollZoom": False,
    "doubleClick": False,
}

# Paul Tol "dark" qualitative palette accents for progress indicators
# (axis colors themselves come from secrets: assessment.axis_colors).
GROWTH_POSITIVE_COLOR = "#225522"
GROWTH_NEGATIVE_COLOR = "#663333"
GROWTH_NEUTRAL_COLOR = "#555555"
PAST_BAR_COLOR = "#BFBFBF"


@st.cache_data(ttl=60)
def get_master_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetches and caches master response data for 60 seconds.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Cached people, responses, q_responses (all latest-only), plus the full
        submissions and q_history frames for longitudinal tracking.
    """
    return load_data()


def _fmt_date(ts) -> str:
    """Formats a submission timestamp compactly (e.g. '5 Mar '25')."""
    if pd.isna(ts):
        return "Date unknown"
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts.strftime('%b')} '{ts.strftime('%y')}"


def _first_name(name) -> str:
    """Returns only the first token of a member's name, for privacy masking.

    Args:
        name: Full display name from the sheet.

    Returns:
        str: First name (or the whole value when it has no spaces).
    """
    parts = str(name).split()
    return parts[0] if parts else "Member"


def _legend_axis_html(axis_name: str, legend_df: pd.DataFrame) -> str:
    """Builds the HTML block for a single axis of the question legend.

    Args:
        axis_name (str): Axis to render (header plus its questions).
        legend_df (pd.DataFrame): Deduplicated Q-number/question/axis rows.

    Returns:
        str: HTML block with underlined axis header and chip-numbered question rows.
    """
    color = AXIS_COLORS.get(axis_name, "#1F77B4")
    parts = [
        f"<div style='font-weight:700; color:{color}; font-size:0.8rem; "
        "letter-spacing:0.05em; text-transform:uppercase; "
        f"border-bottom:2px solid {color}; padding-bottom:3px; margin:0 0 8px'>"
        f"{html.escape(clean_axis_label(axis_name))}</div>"
    ]
    for _, q in legend_df[legend_df["axis"] == axis_name].iterrows():
        parts.append(
            "<div style='display:flex; align-items:flex-start; gap:8px; margin-bottom:6px'>"
            f"<span style='flex:0 0 30px; box-sizing:border-box; text-align:center; "
            f"background:{color}; color:#fff; border-radius:10px; "
            "font-size:0.7rem; font-weight:700; padding:2px 0; margin-top:1px'>"
            f"{html.escape(str(q['q_num']))}</span>"
            f"<span style='font-size:0.85rem; line-height:1.45'>"
            f"{html.escape(str(q['question']))}</span>"
            "</div>"
        )
    return "".join(parts)


def _delta_badge_html(delta: float) -> str:
    """Builds a colored growth badge: green '+2', red '-1', gray '±0'.

    Colors come from the Paul Tol "dark" qualitative palette.

    Args:
        delta (float): Latest score minus previous-take score for a question.

    Returns:
        str: HTML pill showing the signed change.
    """
    if delta > 0:
        bg, label = GROWTH_POSITIVE_COLOR, f"+{delta:g}"
    elif delta < 0:
        bg, label = GROWTH_NEGATIVE_COLOR, f"{delta:g}"
    else:
        bg, label = GROWTH_NEUTRAL_COLOR, "±0"
    return (
        "<span style='flex:0 0 auto; font-size:0.72rem; font-weight:700; "
        f"padding:1px 8px; border-radius:10px; background:{bg}; color:#FFFFFF'>"
        f"{label}</span>"
    )


def _question_header_html(q_num: str, question: str, color: str, trailing_html: str = "") -> str:
    """Builds a question row header: axis-colored Q chip, text, optional trailing badge.

    Args:
        q_num (str): Question label (e.g. 'Q5').
        question (str): Full question text.
        color (str): Axis color for the chip.
        trailing_html (str): Optional right-aligned HTML (delta badge, net summary).

    Returns:
        str: HTML header row.
    """
    return (
        "<div style='display:flex; align-items:flex-start; gap:8px; margin-bottom:4px'>"
        f"<span style='flex:0 0 30px; box-sizing:border-box; text-align:center; "
        f"background:{color}; color:#fff; border-radius:10px; "
        "font-size:0.7rem; font-weight:700; padding:2px 0; margin-top:1px'>"
        f"{html.escape(str(q_num))}</span>"
        f"<span style='flex:1; font-size:0.85rem; line-height:1.45'>"
        f"{html.escape(str(question))}</span>"
        f"{trailing_html}"
        "</div>"
    )


def _score_bar_row_html(label: str, score: float, bar_color: str) -> str:
    """Builds one labeled score bar (date label, 1-5 fill bar, numeric score).

    Args:
        label (str): Left-side label, typically the submission date.
        score (float): Likert score (1-5) setting the fill width.
        bar_color (str): Fill color (axis color for latest, gray for past takes).

    Returns:
        str: HTML bar row, indented to align under the question text.
    """
    pct = max(0.0, min(1.0, float(score) / 5.0)) * 100
    return (
        "<div style='display:flex; align-items:center; gap:8px; margin:3px 0 3px 38px'>"
        f"<span style='flex:0 0 86px; font-size:0.72rem; color:inherit; opacity:0.6'>"
        f"{html.escape(label)}</span>"
        "<span style='flex:1; background:rgba(128,128,128,0.22); border-radius:6px; "
        "height:10px; display:block'>"
        f"<span style='display:block; width:{pct:.0f}%; background:{bar_color}; "
        "height:10px; border-radius:6px'></span>"
        "</span>"
        f"<span style='flex:0 0 20px; font-size:0.8rem; font-weight:700; text-align:right'>"
        f"{score:g}</span>"
        "</div>"
    )


def render_question_key(own_q_responses: pd.DataFrame) -> None:
    """Renders the compact axis map plus the full Q1–Q24 question legend.

    Args:
        own_q_responses (pd.DataFrame): The member's question-level responses,
            carrying the question text captured from the sheet headers.
    """
    if own_q_responses.empty or "question" not in own_q_responses.columns:
        return

    legend_df = (
        own_q_responses[["q_int", "q_num", "question", "axis"]]
        .drop_duplicates(subset="q_int")
        .sort_values("q_int")
    )

    st.subheader("Questions")
    half = (len(AXIS_ORDER) + 1) // 2
    for i in range(half):
        col_left, col_right = st.columns(2)
        col_left.markdown(_legend_axis_html(AXIS_ORDER[i], legend_df), unsafe_allow_html=True)
        if i + half < len(AXIS_ORDER):
            col_right.markdown(
                _legend_axis_html(AXIS_ORDER[i + half], legend_df), unsafe_allow_html=True
            )


def render_member_charts(
    member_responses: pd.DataFrame,
    member_q_responses: pd.DataFrame,
    member_name: str,
    group_avg: pd.Series,
    key_prefix: str,
) -> None:
    """Renders the Axis Comparison radar and Response Breakdown dot plot side by side.

    Used by the Individual View. `st.columns(2)` stacks vertically on narrow
    (mobile) screens automatically.

    Args:
        member_responses (pd.DataFrame): Category-level scores for the member.
        member_q_responses (pd.DataFrame): Question-level scores for the member.
        member_name (str): Display name of the member.
        group_avg (pd.Series): Latest-only average scores for the member's group.
        key_prefix (str): Unique prefix for the Plotly chart keys.
    """
    radar_col, dots_col = st.columns(2)

    with radar_col:
        st.markdown("#### Axis Comparison")
        st.plotly_chart(
            polar_scatter_chart(member_responses, member_name, group_avg),
            use_container_width=True,
            key=f"{key_prefix}-radar",
            config=PLOTLY_CONFIG,
        )

    with dots_col:
        st.markdown("#### Response Breakdown")
        st.plotly_chart(
            individual_question_dot_plot(member_q_responses),
            use_container_width=True,
            key=f"{key_prefix}-dots",
            config=PLOTLY_CONFIG,
        )


def _render_axis_progress_expander(
    axis_name: str,
    q_history: pd.DataFrame,
    n_takes: int,
    key_prefix: str,
) -> None:
    """Renders one axis accordion of per-question longitudinal progress.

    Adapts to submission count: single-period bars for 1 take, past-vs-present
    bars with growth badges for 2-3 takes, and net summary + sparkline per
    question for 4+ takes. Growth is always measured against the previous
    take (t-1), not the first take.

    Args:
        axis_name (str): Axis whose questions are rendered.
        q_history (pd.DataFrame): The member's full question-level history.
        n_takes (int): Number of submissions the member has made.
        key_prefix (str): Unique prefix for sparkline chart keys.
    """
    color = AXIS_COLORS.get(axis_name, "#1F77B4")
    axis_q = q_history[q_history["axis"] == axis_name]

    with st.expander(clean_axis_label(axis_name)):
        for q_int in sorted(axis_q["q_int"].unique()):
            q_rows = axis_q[axis_q["q_int"] == q_int].sort_values("take")
            question = str(q_rows.iloc[-1]["question"])
            q_num = str(q_rows.iloc[-1]["q_num"])
            scores = [float(s) for s in q_rows["score"]]
            dates = [_fmt_date(t) for t in q_rows["timestamp"]]

            if n_takes == 1:
                header = _question_header_html(q_num, question, color)
                bar = _score_bar_row_html(dates[0], scores[0], color)
                st.markdown(
                    f"<div style='margin-bottom:14px'>{header}{bar}</div>",
                    unsafe_allow_html=True,
                )
            elif n_takes <= 3:
                delta = scores[-1] - scores[-2]
                header = _question_header_html(q_num, question, color, _delta_badge_html(delta))
                bars = "".join(
                    _score_bar_row_html(d, s, color if j == len(scores) - 1 else PAST_BAR_COLOR)
                    for j, (d, s) in enumerate(zip(dates, scores))
                )
                st.markdown(
                    f"<div style='margin-bottom:14px'>{header}{bars}</div>",
                    unsafe_allow_html=True,
                )
            else:
                delta = scores[-1] - scores[-2]
                net_summary = (
                    "<span style='flex:0 0 auto; font-size:0.78rem; white-space:nowrap; "
                    f"margin-right:6px'><span style='opacity:0.6'>{scores[-2]:g}</span>"
                    f" ➔ <b>{scores[-1]:g}</b></span>"
                )
                header = _question_header_html(
                    q_num, question, color, net_summary + _delta_badge_html(delta)
                )
                st.markdown(f"<div style='margin-bottom:2px'>{header}</div>", unsafe_allow_html=True)
                st.plotly_chart(
                    question_trend_sparkline(dates, scores, color),
                    use_container_width=True,
                    key=f"{key_prefix}-spark-{q_num}",
                    config=PLOTLY_CONFIG,
                )


def render_progress_section(
    submissions: pd.DataFrame,
    q_history: pd.DataFrame,
    key_prefix: str,
    heading: str = "### Your Progress Over Time",
) -> None:
    """Renders longitudinal per-question progress for one member.

    Questions are grouped by axis into collapsible accordions laid out in two
    columns; Streamlit stacks the columns vertically on mobile, so phones get
    a single scrollable column and desktops get two side-by-side columns.

    Args:
        submissions (pd.DataFrame): The member's submissions, chronological.
        q_history (pd.DataFrame): The member's question-level history.
        key_prefix (str): Unique prefix for chart keys (per view).
        heading (str): Section heading markdown.
    """
    n_takes = len(submissions)
    if n_takes == 0 or q_history.empty:
        return

    st.markdown(heading)
    first_date = _fmt_date(submissions.iloc[0]["timestamp"])
    latest_date = _fmt_date(submissions.iloc[-1]["timestamp"])

    if n_takes == 1:
        st.info(
            "🌱 This is your starting baseline. Take the assessment again in the "
            "future with the **same Member ID** (same name + PIN) and your growth "
            "over time will appear here."
        )
    elif n_takes <= 3:
        st.caption(
            f"**{n_takes} submissions** so far ({first_date} ➔ {latest_date}). "
            "Gray bars = earlier takes · colored bar = latest take. "
            "Badge = change since your previous take."
        )
    else:
        st.caption(
            f"**{n_takes} submissions** so far ({first_date} ➔ {latest_date}). "
            "Each question shows its score trend across dates. "
            "Badge = change since your previous take."
        )

    half = (len(AXIS_ORDER) + 1) // 2
    for i in range(half):
        col_left, col_right = st.columns(2)
        with col_left:
            _render_axis_progress_expander(AXIS_ORDER[i], q_history, n_takes, key_prefix)
        if i + half < len(AXIS_ORDER):
            with col_right:
                _render_axis_progress_expander(AXIS_ORDER[i + half], q_history, n_takes, key_prefix)


def _duplicate_id_warnings(members_df: pd.DataFrame) -> list[str]:
    """Flags members sharing one name under different Member IDs.

    A Member ID is name + PIN, so the same person re-registering with a
    different PIN (or a name typo) creates a duplicate ID whose submissions
    will not link into one history.

    Args:
        members_df (pd.DataFrame): Members to check (name + person_id columns).

    Returns:
        list[str]: One warning line per duplicated name.
    """
    norm = members_df.copy()
    norm["norm_name"] = norm["name"].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    warnings = []
    for _, grp in norm.groupby("norm_name"):
        ids = sorted(grp["person_id"].unique())
        if len(ids) > 1:
            warnings.append(
                f"**{grp['name'].iloc[0]}** has {len(ids)} different IDs: "
                + ", ".join(f"`{i}`" for i in ids)
            )
    return warnings


def individual_view() -> None:
    """Renders Individual View with Member ID authentication, dual plots, and
    longitudinal progress tracking."""
    status = st.empty()
    user_id = st.text_input("Enter your Member ID (e.g. John1234):", key="ind_user_id").strip()

    if not user_id:
        status.info("Please enter your Member ID to access your personal assessment results.")
        return

    try:
        people, responses, q_responses, submissions, q_history = get_master_data()
    except Exception as e:
        status.error(f"Error loading sheet data: {e}")
        return

    matches = people[people["person_id"].str.lower() == user_id.lower()]
    if matches.empty:
        status.error("Member ID not found. Please double-check your ID or name/PIN combination.")
        return

    person = matches.iloc[0]
    own_responses = responses[responses["person_id"] == person["person_id"]]
    own_q_responses = q_responses[q_responses["person_id"] == person["person_id"]]

    if own_responses.empty:
        status.error("No assessment data found for this Member ID.")
        return

    name = person["name"]
    track = "Follower of Jesus" if person["is_believer"] else "Exploring Faith"
    group_avg = category_means(group_responses(people, responses, person["group"]))

    status.success(
        f"Welcome back, **{name}**! | Group: **{person['group']}** | Track: **{track}**"
    )

    st.divider()
    render_member_charts(own_responses, own_q_responses, name, group_avg, key_prefix="chart-ind")

    st.divider()
    own_subs, own_q_hist = person_history(submissions, q_history, person["person_id"])
    render_progress_section(own_subs, own_q_hist, key_prefix="ind")


def leader_view() -> None:
    """Renders Group Leader View with track filtering, member inspect with
    historical snapshots, and Member Directory Table.

    Members who declined leader-sharing consent are excluded from every chart
    and selector; they appear only as masked rows in the directory.
    """
    status = st.empty()
    passcode_input = st.text_input(
        "Enter Season Access Code:", type="password", key="season_passcode_input"
    ).strip()

    passcodes = st.secrets.get("passcodes", {})
    season_groups = st.secrets.get("season_groups", {})

    if not passcode_input or passcode_input not in passcodes:
        status.warning("Please enter a valid Season Access Code to view group summaries.")
        return

    season_code = passcodes[passcode_input]
    allowed_groups = season_groups.get(season_code, [])

    if not allowed_groups:
        status.error(f"No group list configured in secrets for season: **{season_code}**.")
        return

    try:
        people, responses, _, submissions, q_history = get_master_data()
    except Exception as e:
        status.error(f"Error loading master dataset: {e}")
        return

    season_people = people[people["group"].isin(allowed_groups)].reset_index(drop=True)
    season_responses = responses[responses["group"].isin(allowed_groups)].reset_index(drop=True)
    season_submissions = submissions[
        submissions["person_id"].isin(season_people["person_id"])
    ].reset_index(drop=True)
    season_q_history = q_history[
        q_history["person_id"].isin(season_people["person_id"])
    ].reset_index(drop=True)

    if season_people.empty:
        status.info(f"Access Granted for **{season_code}**, but no submissions exist yet.")
        return

    status.success(f"Access Granted for **{season_code}**!")

    col1, col2 = st.columns(2)
    with col1:
        available_groups = [g for g in allowed_groups if g in season_people["group"].values]
        group = st.selectbox("Select Group:", available_groups)

    with col2:
        track_filter = st.selectbox(
            "Filter Track Type:",
            options=["All Tracks", "Believers (Followers)", "Seekers (Exploring)"],
        )

    all_members = season_people[season_people["group"] == group].reset_index(drop=True)

    if track_filter == "Believers (Followers)":
        filtered_members = all_members[all_members["is_believer"] == True].reset_index(drop=True)
    elif track_filter == "Seekers (Exploring)":
        filtered_members = all_members[all_members["is_believer"] == False].reset_index(drop=True)
    else:
        filtered_members = all_members

    if filtered_members.empty:
        st.info("No participants found matching this filter selection.")
        return

    private_count = int((~filtered_members["has_consent"].astype(bool)).sum())
    private_note = ""
    if private_count:
        private_note = (
            f", {private_count} of which "
            + ("chooses" if private_count == 1 else "choose")
            + " to keep their results private"
        )
    st.caption(
        f"Showing **{len(filtered_members)}** members in group **{group}**{private_note} "
        "| 📌 **Solid lines** = Followers | 📌 **Dashed lines** = Exploring Faith"
    )

    # Group averages and trajectories cover consenting members only. Otherwise
    # a leader who sees the group average plus every consenting member's scores
    # could back out a private member's numbers arithmetically in a small group.
    consenting_people = season_people[season_people["has_consent"].astype(bool)]
    group_resp = group_responses(consenting_people, season_responses, group)

    # Members who declined consent are absent from every chart and selector
    # below; they still appear (masked) in the directory at the bottom.
    visible_members = filtered_members[
        filtered_members["has_consent"].astype(bool)
    ].reset_index(drop=True)

    st.divider()
    st.markdown("### Subgroup Chart")

    if visible_members.empty:
        st.info("No member in this selection has consented to share their results with leaders.")
    else:
        pill_options = ["🔍 All Members"] + [
            f"{get_dynamic_color_and_emoji(i, len(visible_members))[1]} {name}"
            for i, name in enumerate(visible_members["name"])
        ]
        selected_pill = st.pills("Highlight Member:", pill_options, default="🔍 All Members")

        highlight_name = None
        if selected_pill and selected_pill != "🔍 All Members":
            highlight_name = selected_pill.split(" ", 1)[1]

        st.plotly_chart(
            group_radar_chart(group_resp, visible_members, highlight_member=highlight_name),
            use_container_width=True,
            key="chart-group-radar",
            config=PLOTLY_CONFIG,
        )

        st.divider()
        st.markdown("### Individual Member Charts")

        member_records = visible_members.to_dict("records")
        selected_idx = st.selectbox(
            "Select a member to view their individual results:",
            options=range(len(member_records)),
            format_func=lambda i: str(member_records[i]["name"]),
            # Key varies by group/track so the widget is recreated (index 0)
            # when either changes — a fixed key would keep a stale positional
            # selection pointing at a different group's member.
            key=f"leader_member_select_{group}_{track_filter}",
        )
        selected_member = member_records[selected_idx]
        member_id = selected_member["person_id"]

        member_subs, member_q_hist = person_history(season_submissions, season_q_history, member_id)

        if member_subs.empty:
            st.info("No assessment data found for this member.")
        else:
            n_takes = len(member_subs)
            latest_take = int(member_subs["take"].max())

            if n_takes > 1:
                take_labels = {
                    int(row["take"]): f"Take {int(row['take'])} · {_fmt_date(row['timestamp'])}"
                    for _, row in member_subs.iterrows()
                }
                selected_take = st.selectbox(
                    "View submission:",
                    options=[int(t) for t in member_subs["take"]],
                    index=n_takes - 1,
                    format_func=lambda t: take_labels[t] + (" (Latest)" if t == latest_take else ""),
                    key=f"leader_take_select_{member_id}",
                )
            else:
                selected_take = latest_take

            sub_row = member_subs[member_subs["take"] == selected_take].iloc[0]
            snapshot_responses = submission_category_scores(sub_row)
            snapshot_q = member_q_hist[member_q_hist["take"] == selected_take]

            # Track/group context as it was at THAT submission (a member who moved
            # from Exploring Faith to Follower of Jesus keeps their old context).
            track_then = "Follower of Jesus" if sub_row["is_believer"] else "Exploring Faith"
            caption = f"Track: **{track_then}**"
            if n_takes > 1:
                caption += f" | Submission **{selected_take} of {n_takes}**"
            if sub_row["group"] != group:
                caption += f" | Group at that time: **{sub_row['group']}**"
            st.caption(caption)

            # Leaders get one combined chart instead of the individual-vs-group
            # radar: the member's axis averages as a translucent spiderweb
            # overlaid on their per-question breakdown.
            st.markdown("#### Response Breakdown")
            st.plotly_chart(
                individual_question_dot_plot(snapshot_q, axis_scores_df=snapshot_responses),
                use_container_width=True,
                key="chart-leader-ind-dots",
                config=PLOTLY_CONFIG,
            )

            st.divider()
            render_question_key(snapshot_q)

    st.divider()
    st.markdown("### Group Member Directory")
    directory_df = filtered_members[["name", "person_id", "is_believer", "has_consent"]].copy()
    directory_df["Track"] = directory_df["is_believer"].map(
        {True: "Follower of Jesus", False: "Exploring Faith"}
    )
    # Count questionnaires while the real IDs are still present.
    take_counts = season_submissions.groupby("person_id").size()
    directory_df["Takes"] = directory_df["person_id"].map(take_counts).fillna(0).astype(int)

    # Members who declined consent keep only First Name, Track and questionnaire
    # count; their Unique ID is shown as "Private".
    private_mask = ~directory_df["has_consent"].astype(bool)
    directory_df.loc[private_mask, "name"] = directory_df.loc[private_mask, "name"].map(_first_name)
    directory_df.loc[private_mask, "person_id"] = "Private"

    directory_df.rename(
        columns={"name": "Member Name", "person_id": "Member ID"}, inplace=True
    )

    st.dataframe(
        directory_df[["Member Name", "Member ID", "Track", "Takes"]],
        use_container_width=True,
        hide_index=True,
    )

    # Only consenting members are checked, so a private ID is never printed here.
    dup_warnings = _duplicate_id_warnings(all_members[all_members["has_consent"].astype(bool)])
    if dup_warnings:
        st.warning(
            "⚠️ **Possible duplicate Member IDs in this group:**\n\n"
            + "\n".join(f"- {w}" for w in dup_warnings)
            + "\n\nEach ID is counted once (latest submission only), but if one person "
            "holds two IDs their history is split. Ask them to reuse their original "
            "name + PIN so future submissions link to a single ID."
        )


def main() -> None:
    """Main application layout and tab execution."""
    st.title("🕊️ Spiritual Health Assessment")

    tab_ind, tab_lead = st.tabs(["👤 Individual View", "👥 Leader Dashboard"])

    with tab_ind:
        individual_view()

    with tab_lead:
        leader_view()


if __name__ == "__main__":
    main()
