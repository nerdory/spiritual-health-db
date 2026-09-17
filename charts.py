"""Plotly chart generation module for Spiritual Health Assessment.

Renders polar radar plots and distinct angular-spread question dot plots.
"""

import colorsys
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from assessment_config import get_axis_colors, get_axis_order

# Axis order and colors come from secrets (see assessment_config.py);
# the key order of `assessment.axis_mapping` defines the display order.
AXIS_ORDER: list[str] = get_axis_order()
AXIS_COLORS: dict[str, str] = get_axis_colors()


def clean_axis_label(axis_str: str) -> str:
    """Strips numerical prefix from axis label.

    Args:
        axis_str (str): Full axis category name.

    Returns:
        str: Cleaned display name.
    """
    return axis_str.split(":", 1)[1].strip() if ":" in str(axis_str) else str(axis_str)


def _wrap_axis_label(label: str, max_chars: int = 12) -> str:
    """Inserts line breaks into an axis label so it fits on narrow screens.

    Args:
        label (str): Cleaned axis label.
        max_chars (int): Soft maximum characters per line.

    Returns:
        str: Label with <br> inserted at word boundaries.
    """
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return "<br>".join(lines)


WRAPPED_AXIS_LABELS: list[str] = [_wrap_axis_label(clean_axis_label(a)) for a in AXIS_ORDER]


def get_dynamic_color_and_emoji(index: int, total_count: int) -> tuple[str, str]:
    """Generates dynamic colors and matching emojis evenly distributed across hue angles.

    Args:
        index (int): Index of item.
        total_count (int): Total number of items.

    Returns:
        tuple[str, str]: Hex color code and matching emoji.
    """
    if total_count <= 0:
        total_count = 1

    hue = (index / total_count) % 1.0
    r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.85)
    hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    hue_deg = hue * 360
    if hue_deg < 25 or hue_deg >= 345:
        emoji = "🔴"
    elif hue_deg < 50:
        emoji = "🟠"
    elif hue_deg < 70:
        emoji = "🟡"
    elif hue_deg < 165:
        emoji = "🟢"
    elif hue_deg < 195:
        emoji = "🩵"
    elif hue_deg < 260:
        emoji = "🔵"
    elif hue_deg < 315:
        emoji = "🟣"
    else:
        emoji = "💖"

    return hex_color, emoji


def polar_scatter_chart(
    individual_responses_df: pd.DataFrame, member_name: str, group_means: pd.Series
) -> go.Figure:
    """Renders top radar chart comparing an individual member against their group average.

    Args:
        individual_responses_df (pd.DataFrame): Scores for the target member.
        member_name (str): Display name of the member.
        group_means (pd.Series): Deduplicated average scores for the member's group.

    Returns:
        go.Figure: Interactive radar chart figure.
    """
    fig = go.Figure()
    clean_axes = WRAPPED_AXIS_LABELS

    score_dict = dict(zip(individual_responses_df["category"], individual_responses_df["score"]))
    ind_scores = [score_dict.get(a, 0) for a in AXIS_ORDER]

    fig.add_trace(
        go.Scatterpolar(
            r=ind_scores + [ind_scores[0]],
            theta=clean_axes + [clean_axes[0]],
            fill="toself",
            name=f"{member_name}'s Scores",
            line=dict(color="#1F77B4", width=3),
            marker=dict(size=7, color="#1F77B4"),
            fillcolor="rgba(31, 119, 180, 0.2)",
        )
    )

    if not group_means.empty:
        grp_dict = group_means.to_dict()
        grp_scores = [grp_dict.get(a, 0) for a in AXIS_ORDER]

        fig.add_trace(
            go.Scatterpolar(
                r=grp_scores + [grp_scores[0]],
                theta=clean_axes + [clean_axes[0]],
                fill=None,
                mode="lines",
                name="Group Average",
                line=dict(color="#7F7F7F", width=2, dash="dash"),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], dtick=1),
            angularaxis=dict(direction="clockwise"),
        ),
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        dragmode=False,
    )
    return fig


def individual_question_dot_plot(
    member_q_responses: pd.DataFrame, axis_scores_df: pd.DataFrame | None = None
) -> go.Figure:
    """Renders bottom non-overlapping circular dot plot for individual question scores.

    Places every question at its own computed angle: each axis owns a 60°
    sector (with a labeled gridline at its center), and that axis's questions
    are spread symmetrically around the center so no two datapoints ever
    share the same spoke. Radius encodes the exact Likert score.

    Args:
        member_q_responses (pd.DataFrame): Question-level scores (Q1-Q24) for member.
        axis_scores_df (pd.DataFrame | None): Optional category/score frame with
            the member's per-axis averages. When given, a translucent blue
            spiderweb is drawn beneath the dots, connecting each axis average
            at its sector's center gridline.

    Returns:
        go.Figure: Circular dot plot figure.
    """
    fig = go.Figure()

    sector_deg = 360 / len(AXIS_ORDER)  # 60° per axis
    axis_center = {axis: idx * sector_deg for idx, axis in enumerate(AXIS_ORDER)}

    # Spiderweb overlay is added first so the question dots render on top.
    if axis_scores_df is not None and not axis_scores_df.empty:
        score_dict = dict(zip(axis_scores_df["category"], axis_scores_df["score"]))
        web_scores = [score_dict.get(a, 0) for a in AXIS_ORDER]
        web_labels = [clean_axis_label(a) for a in AXIS_ORDER]
        fig.add_trace(
            go.Scatterpolar(
                r=web_scores + [web_scores[0]],
                theta=[axis_center[a] for a in AXIS_ORDER] + [axis_center[AXIS_ORDER[0]]],
                fill="toself",
                mode="lines",
                name="Axis Average",
                line=dict(color="#1F77B4", width=2),
                fillcolor="rgba(31, 119, 180, 0.12)",
                opacity=0.55,
                customdata=[[lbl, f"{s:g}"] for lbl, s in zip(web_labels, web_scores)]
                + [[web_labels[0], f"{web_scores[0]:g}"]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>Axis average: "
                    "<b>%{customdata[1]} / 5</b><extra></extra>"
                ),
            )
        )

    df = member_q_responses.copy()
    df["color"] = df["axis"].map(AXIS_COLORS).fillna("#1F77B4")

    for axis_name in AXIS_ORDER:
        axis_df = df[df["axis"] == axis_name].sort_values("q_int")
        if axis_df.empty:
            continue

        # Spread the axis's questions symmetrically around its center angle.
        # Step is capped so points stay 6° clear of the neighbouring sector.
        n = len(axis_df)
        max_step = (sector_deg - 12) / max(n - 1, 1)
        step = min(11.0, max_step)
        offsets = (np.arange(n) - (n - 1) / 2) * step
        thetas = (axis_center[axis_name] + offsets) % 360

        if "question" in axis_df.columns:
            questions = axis_df["question"].fillna(axis_df["q_num"]).astype(str)
        else:
            questions = axis_df["q_num"].astype(str)
        hover_questions = [(q[:97] + "…") if len(q) > 100 else q for q in questions]
        score_labels = axis_df["score"].map(lambda s: f"{s:g}")

        fig.add_trace(
            go.Scatterpolar(
                r=axis_df["score"],
                theta=thetas,
                mode="markers+text",
                name=clean_axis_label(axis_name),
                text=axis_df["q_int"].astype(str),
                textposition="middle center",
                textfont=dict(color="white", size=9, family="Arial Black"),
                marker=dict(
                    size=22,
                    color=axis_df["color"],
                    line=dict(width=1, color="white"),
                ),
                customdata=list(zip(axis_df["q_num"], score_labels, hover_questions)),
                hovertemplate=(
                    "<b>%{customdata[0]}</b>: %{customdata[2]}"
                    "<br>Score: <b>%{customdata[1]} / 5</b><extra></extra>"
                ),
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5.3], dtick=1),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                tickmode="array",
                tickvals=[axis_center[a] for a in AXIS_ORDER],
                ticktext=WRAPPED_AXIS_LABELS,
                showgrid=True,
                gridcolor="rgba(0, 0, 0, 0.2)",
                tickfont=dict(size=11),
            ),
        ),
        margin=dict(l=60, r=60, t=50, b=50),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center", title="Axes"),
        dragmode=False,
    )
    return fig


def question_trend_sparkline(
    dates: list[str], scores: list[float], color: str = "#1F77B4"
) -> go.Figure:
    """Renders a mini trend line (sparkline) of one question's scores over time.

    Used when a member has 4+ submissions: shows score progression across
    submission dates on the fixed 1–5 Likert scale, with details on hover.

    Args:
        dates (list[str]): Display labels for each submission (chronological).
        scores (list[float]): Score per submission, aligned with `dates`.
        color (str): Line/marker color (typically the question's axis color).

    Returns:
        go.Figure: Compact line chart figure.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=scores,
            mode="lines+markers",
            line=dict(color=color, width=2, shape="spline", smoothing=0.6),
            marker=dict(size=7, color=color, line=dict(width=1, color="white")),
            hovertemplate="%{x}<br>Score: <b>%{y:g} / 5</b><extra></extra>",
        )
    )

    fig.update_layout(
        height=110,
        margin=dict(l=34, r=8, t=6, b=24),
        xaxis=dict(
            tickfont=dict(size=9, color="rgba(0,0,0,0.55)"),
            showgrid=False,
            zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[0.5, 5.5],
            dtick=1,
            tickfont=dict(size=9, color="rgba(0,0,0,0.55)"),
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            fixedrange=True,
        ),
        showlegend=False,
        dragmode=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def group_radar_chart(
    group_responses_df: pd.DataFrame,
    members_df: pd.DataFrame,
    highlight_member: str | None = None,
) -> go.Figure:
    """Renders group radar chart showing member trajectories.

    Callers pass consenting members only, so colors and legend entries align
    with the highlight pills. Members with `has_consent = False` are skipped
    here too, as a defensive measure.

    Args:
        group_responses_df (pd.DataFrame): Responses for members in group.
        members_df (pd.DataFrame): Member details DataFrame.
        highlight_member (str | None): Member name to visually isolate.

    Returns:
        go.Figure: Comparative group radar chart.
    """
    fig = go.Figure()
    clean_axes = WRAPPED_AXIS_LABELS
    total_members = len(members_df)

    for idx, row in members_df.iterrows():
        member_name = row["name"]
        is_believer = row.get("is_believer", True)

        if not bool(row.get("has_consent", True)):
            continue

        member_resp = group_responses_df[group_responses_df["name"] == member_name]

        if member_resp.empty:
            continue

        score_dict = dict(zip(member_resp["category"], member_resp["score"]))
        scores = [score_dict.get(a, 0) for a in AXIS_ORDER]

        color, emoji = get_dynamic_color_and_emoji(idx, total_members)
        line_dash = "solid" if is_believer else "dash"

        if highlight_member:
            if member_name == highlight_member:
                opacity, line_width = 1.0, 4.0
            else:
                opacity, line_width = 0.12, 1.2
        else:
            opacity, line_width = 0.8, 2.2

        fig.add_trace(
            go.Scatterpolar(
                r=scores + [scores[0]],
                theta=clean_axes + [clean_axes[0]],
                mode="lines+markers",
                name=f"{emoji} {member_name}",
                line=dict(color=color, width=line_width, dash=line_dash),
                marker=dict(size=6, color=color),
                opacity=opacity,
                hovertemplate=f"<b>{member_name}</b><br>%{{theta}}: %{{r:.2f}}<extra></extra>",
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 5], dtick=1),
            angularaxis=dict(direction="clockwise"),
        ),
        margin=dict(l=60, r=60, t=40, b=40),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
        dragmode=False,
    )
    return fig
