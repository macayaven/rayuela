#!/usr/bin/env python3
"""
Article-ready visualizations for Project Rayuela.

Generates publication-quality figures for the Medium article, covering:
  1. Narrative DNA heatmap (chapters × 20 dimensions)
  2. Reading path emotional journeys (sparklines per dimension)
  3. Trajectory smoothness comparison (linear vs hopscotch)
  4. Chapter fingerprint radar charts (notable chapters)
  5. Dimension correlation matrix
  6. Section box plots (dimension distributions by section)

Usage (inside Docker container):
    python src/article_figures.py

    # Specific figure only:
    python src/article_figures.py --only heatmap

Output: outputs/figures/article_*.html
"""

import argparse
import json
import numpy as np
from pathlib import Path
from itertools import pairwise

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"

# Cortázar's Tablero de Dirección (hopscotch reading order)
TABLERO = [
    73, 1, 2, 116, 3, 84, 4, 71, 5, 81, 74, 6, 7, 8, 93, 68, 9, 104,
    10, 65, 11, 136, 12, 106, 13, 115, 14, 114, 117, 15, 120, 16, 137,
    17, 97, 18, 153, 19, 90, 20, 126, 21, 79, 22, 62, 23, 124, 128, 24,
    134, 25, 141, 60, 26, 109, 27, 28, 130, 151, 152, 131, 29, 139, 30,
    138, 31, 32, 132, 33, 140, 34, 135, 35, 105, 36, 63, 37, 98, 38,
    102, 39, 113, 40, 120, 41, 100, 42, 76, 43, 44, 108, 45, 69, 46,
    101, 47, 110, 48, 111, 49, 118, 50, 119, 51, 69, 52, 89, 53, 66,
    149, 54, 129, 139, 133, 140, 138, 127, 56, 135, 63, 88, 72, 77, 131,
    58, 131,
]

LINEAR_ORDER = list(range(1, 57))

SECTION_COLORS = {
    "Del lado de allá": "#2196F3",
    "Del lado de acá": "#FF9800",
    "De otros lados (Capítulos prescindibles)": "#9C27B0",
}
SECTION_SHORT = {
    "Del lado de allá": "Allá (Paris)",
    "Del lado de acá": "Acá (Buenos Aires)",
    "De otros lados (Capítulos prescindibles)": "Otros lados (Expendable)",
}

# Dimension grouping for visualization
DIM_GROUPS = {
    "Thematic": [
        "existential_questioning", "art_and_aesthetics",
        "everyday_mundanity", "death_and_mortality", "love_and_desire",
    ],
    "Emotional": [
        "emotional_intensity", "humor_and_irony",
        "melancholy_and_nostalgia", "tension_and_anxiety",
    ],
    "Character": [
        "oliveira_centrality", "la_maga_presence",
        "character_density", "interpersonal_conflict",
    ],
    "Narrative": [
        "interiority", "dialogue_density", "metafiction", "temporal_clarity",
    ],
    "Formal": [
        "spatial_grounding", "language_experimentation", "intertextual_density",
    ],
}

# Flat ordered list matching the grouping
DIMS_ORDERED = [d for dims in DIM_GROUPS.values() for d in dims]

# Pretty display names for dimensions
DIM_LABELS = {
    "existential_questioning": "Existential",
    "art_and_aesthetics": "Art/Aesthetics",
    "everyday_mundanity": "Mundanity",
    "death_and_mortality": "Death",
    "love_and_desire": "Love/Desire",
    "emotional_intensity": "Emotion",
    "humor_and_irony": "Humor/Irony",
    "melancholy_and_nostalgia": "Melancholy",
    "tension_and_anxiety": "Tension",
    "oliveira_centrality": "Oliveira",
    "la_maga_presence": "La Maga",
    "character_density": "Characters",
    "interpersonal_conflict": "Conflict",
    "interiority": "Interiority",
    "dialogue_density": "Dialogue",
    "metafiction": "Metafiction",
    "temporal_clarity": "Temporal frag.",
    "spatial_grounding": "Spatial",
    "language_experimentation": "Lang. experiment",
    "intertextual_density": "Intertextual",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load chapter metadata and narrative DNA vectors."""
    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    chapters_meta = raw["chapters"]

    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        dna = json.load(f)
    dna_chapters = dna["chapters"]
    dims = dna["dimensions"]

    # Build score matrix (N_extracted × 20) and chapter number list
    ch_nums = [ch["chapter"] for ch in dna_chapters]
    scores = {
        ch["chapter"]: {d: ch["scores"][d] for d in dims}
        for ch in dna_chapters
    }

    # Build metadata lookup
    meta_by_num = {ch["number"]: ch for ch in chapters_meta}

    # Load Scale A embeddings
    emb_a = np.load(PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy")

    return chapters_meta, dna_chapters, dims, scores, meta_by_num, emb_a


# ---------------------------------------------------------------------------
# 1. Narrative DNA Heatmap
# ---------------------------------------------------------------------------

def fig_heatmap(dna_chapters, dims, scores, meta_by_num):
    """
    Heatmap: chapters (rows) × 20 dimensions (columns).

    Rows ordered by chapter number, with section color annotation.
    Columns grouped by dimension category.
    """
    # Order chapters by number
    ch_nums = sorted(scores.keys())
    n = len(ch_nums)

    # Build matrix using grouped dimension order
    matrix = np.array([
        [scores[ch][d] for d in DIMS_ORDERED]
        for ch in ch_nums
    ])

    # Section color strip
    section_colors = []
    for ch in ch_nums:
        sec = meta_by_num[ch]["section"]
        section_colors.append(SECTION_COLORS.get(sec, "#999"))

    # Y-axis labels
    y_labels = [f"Ch.{ch}" for ch in ch_nums]
    x_labels = [DIM_LABELS.get(d, d) for d in DIMS_ORDERED]

    # Custom hover text
    hover = []
    for i, ch in enumerate(ch_nums):
        row = []
        for j, d in enumerate(DIMS_ORDERED):
            row.append(
                f"Ch.{ch} — {DIM_LABELS.get(d, d)}: {scores[ch][d]}"
            )
        hover.append(row)

    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=matrix,
        x=x_labels,
        y=y_labels,
        colorscale="RdYlBu_r",  # Red=high, Blue=low
        zmin=1, zmax=10,
        text=hover,
        hoverinfo="text",
        colorbar=dict(title="Score", x=1.02),
    ))

    # Add section color strip as a narrow heatmap on the left
    # (encoded as numeric: 0=Allá, 1=Acá, 2=Otros)
    sec_map = {"Del lado de allá": 0, "Del lado de acá": 1,
               "De otros lados (Capítulos prescindibles)": 2}
    sec_vals = [[sec_map.get(meta_by_num[ch]["section"], 2)] for ch in ch_nums]

    # Add dimension group separators as vertical lines
    group_boundaries = []
    cumulative = 0
    for group_name, group_dims in DIM_GROUPS.items():
        cumulative += len(group_dims)
        group_boundaries.append((cumulative - 0.5, group_name))

    for boundary, _ in group_boundaries[:-1]:  # Skip last (end of chart)
        fig.add_vline(x=boundary, line_width=2, line_color="white")

    # Group labels as annotations at the top
    cumulative = 0
    for group_name, group_dims in DIM_GROUPS.items():
        mid = cumulative + len(group_dims) / 2 - 0.5
        fig.add_annotation(
            x=mid, y=-0.5,
            text=f"<b>{group_name}</b>",
            showarrow=False,
            font=dict(size=10, color="#555"),
            yref="y",
            yshift=15,
        )
        cumulative += len(group_dims)

    fig.update_layout(
        title=dict(
            text="Narrative DNA: 20-Dimensional Profile of Each Chapter",
            font=dict(size=16),
        ),
        xaxis=dict(
            title="", side="top", tickangle=-45,
            tickfont=dict(size=9),
        ),
        yaxis=dict(
            title="", autorange="reversed",
            tickfont=dict(size=7),
            dtick=5,
        ),
        width=900,
        height=max(600, n * 5 + 100),
        margin=dict(l=60, r=80, t=120, b=40),
        plot_bgcolor="white",
    )

    return fig


# ---------------------------------------------------------------------------
# 2. Reading Path Emotional Journeys
# ---------------------------------------------------------------------------

def fig_emotional_journeys(scores, meta_by_num):
    """
    Multi-line sparklines showing how key dimensions evolve along
    the linear and hopscotch reading paths.

    Shows the "emotional arc" of each reading experience.
    """
    # Select the most narratively interesting dimensions
    journey_dims = [
        "emotional_intensity", "tension_and_anxiety", "humor_and_irony",
        "love_and_desire", "existential_questioning", "melancholy_and_nostalgia",
    ]

    dim_colors = {
        "emotional_intensity": "#E53935",
        "tension_and_anxiety": "#FF6F00",
        "humor_and_irony": "#43A047",
        "love_and_desire": "#E91E63",
        "existential_questioning": "#5C6BC0",
        "melancholy_and_nostalgia": "#78909C",
    }

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Linear Reading (Ch. 1→56): Emotional Journey",
            "Hopscotch Reading (Tablero): Emotional Journey",
        ),
        vertical_spacing=0.12,
        shared_xaxes=False,
    )

    for row, (path, path_name) in enumerate([
        (LINEAR_ORDER, "Linear"),
        (TABLERO, "Hopscotch"),
    ], start=1):
        # Filter to chapters with scores
        valid_path = [ch for ch in path if ch in scores]

        for dim in journey_dims:
            values = [scores[ch][dim] for ch in valid_path]

            # Apply light smoothing (3-point moving average) for readability
            smoothed = []
            for i in range(len(values)):
                window = values[max(0, i-1):i+2]
                smoothed.append(sum(window) / len(window))

            fig.add_trace(go.Scatter(
                x=list(range(len(valid_path))),
                y=smoothed,
                mode="lines",
                name=DIM_LABELS.get(dim, dim),
                line=dict(color=dim_colors.get(dim, "#999"), width=1.5),
                opacity=0.8,
                showlegend=(row == 1),
                hovertemplate=(
                    f"<b>{DIM_LABELS.get(dim, dim)}</b><br>"
                    "Step %{x}: Ch.%{customdata}<br>"
                    "Score: %{y:.1f}<extra></extra>"
                ),
                customdata=[str(ch) for ch in valid_path],
            ), row=row, col=1)

        # Add chapter number annotations for key moments
        # (every 10th step)
        for i in range(0, len(valid_path), 10):
            fig.add_annotation(
                x=i, y=0.5,
                text=f"Ch.{valid_path[i]}",
                showarrow=False,
                font=dict(size=7, color="#999"),
                row=row, col=1,
                yref=f"y{row}" if row > 1 else "y",
            )

    fig.update_layout(
        title="The Two Reading Experiences: Emotional Arcs Compared",
        width=1100,
        height=700,
        plot_bgcolor="white",
        legend=dict(x=1.02, y=1, font=dict(size=10)),
    )
    fig.update_xaxes(title_text="Reading step", row=2, col=1)
    fig.update_yaxes(title_text="Score (1-10)", range=[0.5, 10.5])

    return fig


# ---------------------------------------------------------------------------
# 3. Trajectory Smoothness Comparison
# ---------------------------------------------------------------------------

def fig_trajectory_smoothness(scores, emb_a, chapters_meta):
    """
    Compare step-by-step "jumps" in embedding space for the two reading paths.

    For each consecutive pair in a reading order, compute the distance in both
    Scale A (texture) and Scale B (narrative) space. Plot as histograms and
    cumulative curves.
    """
    ch_to_a_idx = {ch["number"]: i for i, ch in enumerate(chapters_meta)}
    dims_list = DIMS_ORDERED

    def path_distances(path, space="A"):
        """Compute consecutive distances along a reading path."""
        dists = []
        for ch_a, ch_b in pairwise(path):
            if space == "A":
                if ch_a in ch_to_a_idx and ch_b in ch_to_a_idx:
                    va = emb_a[ch_to_a_idx[ch_a]]
                    vb = emb_a[ch_to_a_idx[ch_b]]
                    # Cosine distance
                    cos_sim = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                    dists.append(1 - cos_sim)
            elif space == "B":
                if ch_a in scores and ch_b in scores:
                    va = np.array([scores[ch_a][d] for d in dims_list])
                    vb = np.array([scores[ch_b][d] for d in dims_list])
                    dists.append(np.linalg.norm(va - vb))
        return dists

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Scale A (Texture): Step-by-Step Distance",
            "Scale B (Narrative): Step-by-Step Distance",
            "Scale A: Distance Distribution",
            "Scale B: Distance Distribution",
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )

    for col, space in enumerate(["A", "B"], start=1):
        linear_dists = path_distances(LINEAR_ORDER, space)
        hop_dists = path_distances(TABLERO, space)

        # Step-by-step line plot (top row)
        fig.add_trace(go.Scatter(
            x=list(range(len(linear_dists))),
            y=linear_dists,
            mode="lines",
            name=f"Linear ({space})",
            line=dict(color="#4CAF50", width=1.5),
            opacity=0.7,
            showlegend=(col == 1),
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=list(range(len(hop_dists))),
            y=hop_dists,
            mode="lines",
            name=f"Hopscotch ({space})",
            line=dict(color="#F44336", width=1),
            opacity=0.5,
            showlegend=(col == 1),
        ), row=1, col=col)

        # Histogram (bottom row)
        fig.add_trace(go.Histogram(
            x=linear_dists,
            name="Linear",
            marker_color="rgba(76, 175, 80, 0.5)",
            nbinsx=25,
            showlegend=False,
        ), row=2, col=col)

        fig.add_trace(go.Histogram(
            x=hop_dists,
            name="Hopscotch",
            marker_color="rgba(244, 67, 54, 0.5)",
            nbinsx=25,
            showlegend=False,
        ), row=2, col=col)

        # Add mean lines
        if linear_dists:
            fig.add_vline(
                x=np.mean(linear_dists), row=2, col=col,
                line=dict(color="#4CAF50", width=2, dash="dash"),
                annotation_text=f"μ={np.mean(linear_dists):.3f}",
                annotation_font_size=9,
            )
        if hop_dists:
            fig.add_vline(
                x=np.mean(hop_dists), row=2, col=col,
                line=dict(color="#F44336", width=2, dash="dash"),
                annotation_text=f"μ={np.mean(hop_dists):.3f}",
                annotation_font_size=9,
            )

    fig.update_layout(
        title="Trajectory Smoothness: How 'Jumpy' Is Each Reading Path?",
        width=1100,
        height=800,
        plot_bgcolor="white",
        barmode="overlay",
    )
    fig.update_xaxes(title_text="Reading step", row=1)
    fig.update_xaxes(title_text="Distance between consecutive chapters", row=2)
    fig.update_yaxes(title_text="Distance", row=1)
    fig.update_yaxes(title_text="Count", row=2)

    return fig


# ---------------------------------------------------------------------------
# 4. Chapter Fingerprint Radar Charts
# ---------------------------------------------------------------------------

def fig_radar_fingerprints(scores):
    """
    Radar (polar) charts comparing the 20-dimensional profiles of
    notable chapters — the "narrative fingerprints."
    """
    # Notable chapters with descriptions
    notable = {
        1: "Ch.1 — Opening (Oliveira meets La Maga)",
        7: "Ch.7 — Club de la Serpiente debate",
        28: "Ch.28 — Berthe Trépat concert",
        36: "Ch.36 — Oliveira on the bridge",
        56: "Ch.56 — End of Part II (circus scene)",
        68: "Ch.68 — Glíglico (invented language of love)",
        73: "Ch.73 — First in hopscotch order (Morelli)",
        93: "Ch.93 — Morelli literary theory",
        34: "Ch.34 — Two-column interleaved chapter",
        155: "Ch.155 — Final expendable chapter",
    }

    # Filter to chapters that have scores
    available = {ch: desc for ch, desc in notable.items() if ch in scores}

    if len(available) < 3:
        print(f"  Only {len(available)} notable chapters available, need ≥3")
        return None

    labels = [DIM_LABELS.get(d, d) for d in DIMS_ORDERED]

    fig = go.Figure()

    colors = px.colors.qualitative.Set2
    for i, (ch_num, desc) in enumerate(available.items()):
        values = [scores[ch_num][d] for d in DIMS_ORDERED]
        values.append(values[0])  # Close the polygon

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=desc,
            opacity=0.3,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title="Narrative Fingerprints: Profiles of Notable Chapters",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], tickfont=dict(size=8)),
            angularaxis=dict(tickfont=dict(size=8)),
        ),
        width=900,
        height=700,
        legend=dict(x=1.05, y=1, font=dict(size=9)),
    )

    return fig


# ---------------------------------------------------------------------------
# 5. Dimension Correlation Matrix
# ---------------------------------------------------------------------------

def fig_correlation_matrix(scores):
    """
    Correlation matrix of the 20 narrative dimensions across all chapters.

    Reveals which dimensions co-occur — the "grammar" of Cortázar's writing modes.
    """
    ch_nums = sorted(scores.keys())
    matrix = np.array([
        [scores[ch][d] for d in DIMS_ORDERED]
        for ch in ch_nums
    ])

    corr = np.corrcoef(matrix.T)
    labels = [DIM_LABELS.get(d, d) for d in DIMS_ORDERED]

    # Custom hover text
    hover = []
    for i, l1 in enumerate(labels):
        row = []
        for j, l2 in enumerate(labels):
            row.append(f"{l1} × {l2}: r = {corr[i, j]:.2f}")
        hover.append(row)

    fig = go.Figure(go.Heatmap(
        z=corr,
        x=labels,
        y=labels,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=hover,
        hoverinfo="text",
        colorbar=dict(title="Pearson r"),
    ))

    # Add dimension group separators
    cumulative = 0
    for group_name, group_dims in DIM_GROUPS.items():
        cumulative += len(group_dims)
        if cumulative < len(DIMS_ORDERED):
            fig.add_hline(y=cumulative - 0.5, line_width=2, line_color="white")
            fig.add_vline(x=cumulative - 0.5, line_width=2, line_color="white")

    fig.update_layout(
        title="Dimension Correlation Matrix: Which Traits Co-occur?",
        width=800,
        height=750,
        xaxis=dict(tickangle=-45, tickfont=dict(size=9), side="bottom"),
        yaxis=dict(tickfont=dict(size=9), autorange="reversed"),
        margin=dict(l=100, r=80, t=60, b=120),
    )

    return fig


# ---------------------------------------------------------------------------
# 6. Section Box Plots
# ---------------------------------------------------------------------------

def fig_section_distributions(scores, meta_by_num):
    """
    Box plots showing how each dimension distributes across the three sections.

    Answers: What is thematically different about Paris, Buenos Aires,
    and the expendable chapters?
    """
    ch_nums = sorted(scores.keys())

    # Build data for each section
    data_rows = []
    for ch in ch_nums:
        sec = meta_by_num[ch]["section"]
        sec_label = SECTION_SHORT.get(sec, sec)
        for d in DIMS_ORDERED:
            data_rows.append({
                "chapter": ch,
                "section": sec_label,
                "dimension": DIM_LABELS.get(d, d),
                "score": scores[ch][d],
                "dim_key": d,
            })

    # Select dimensions with the most variation between sections
    # (compute F-statistic equivalent: between-group variance / within-group variance)
    import statistics
    dim_interest = {}
    for d in DIMS_ORDERED:
        section_means = {}
        for sec in SECTION_SHORT.values():
            vals = [r["score"] for r in data_rows if r["section"] == sec and r["dim_key"] == d]
            if vals:
                section_means[sec] = statistics.mean(vals)
        if len(section_means) >= 2:
            overall_mean = statistics.mean(section_means.values())
            between_var = statistics.variance(section_means.values()) if len(section_means) > 1 else 0
            dim_interest[d] = between_var

    # Pick top 10 most differentiating dimensions
    top_dims = sorted(dim_interest, key=dim_interest.get, reverse=True)[:10]
    top_labels = [DIM_LABELS.get(d, d) for d in top_dims]

    fig = go.Figure()

    sec_colors = {
        "Allá (Paris)": "#2196F3",
        "Acá (Buenos Aires)": "#FF9800",
        "Otros lados (Expendable)": "#9C27B0",
    }

    for sec_label, color in sec_colors.items():
        for d, d_label in zip(top_dims, top_labels):
            vals = [r["score"] for r in data_rows
                    if r["section"] == sec_label and r["dim_key"] == d]
            if not vals:
                continue
            fig.add_trace(go.Box(
                y=vals,
                x=[d_label] * len(vals),
                name=sec_label,
                marker_color=color,
                legendgroup=sec_label,
                showlegend=(d == top_dims[0]),
                boxmean=True,
            ))

    fig.update_layout(
        title="How Do the Three Sections Differ? (Top 10 Distinguishing Dimensions)",
        xaxis=dict(title="", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(title="Score (1-10)", range=[0.5, 10.5]),
        boxmode="group",
        width=1100,
        height=600,
        plot_bgcolor="white",
        legend=dict(x=0.01, y=0.99),
    )

    return fig


# ---------------------------------------------------------------------------
# 7. Hopscotch vs Linear: Dimension-by-Dimension Journey
# ---------------------------------------------------------------------------

def fig_dual_journey_detail(scores):
    """
    Side-by-side detailed view: one row per key dimension, showing
    how that dimension evolves along both reading paths.

    This is the "definitive" visualization showing that the two reading
    experiences produce genuinely different emotional/thematic arcs.
    """
    key_dims = [
        "emotional_intensity", "tension_and_anxiety", "humor_and_irony",
        "existential_questioning", "love_and_desire", "death_and_mortality",
        "interiority", "metafiction",
    ]

    n_dims = len(key_dims)
    fig = make_subplots(
        rows=n_dims, cols=2,
        subplot_titles=[
            f"Linear: {DIM_LABELS[d]}" if col == 0 else f"Hopscotch: {DIM_LABELS[d]}"
            for d in key_dims for col in [0, 1]
        ],
        vertical_spacing=0.03,
        horizontal_spacing=0.05,
        shared_yaxes=True,
    )

    for row_idx, dim in enumerate(key_dims, start=1):
        for col_idx, (path, color) in enumerate([
            (LINEAR_ORDER, "#4CAF50"),
            (TABLERO, "#F44336"),
        ], start=1):
            valid = [ch for ch in path if ch in scores]
            vals = [scores[ch][dim] for ch in valid]

            # Smoothed version (5-point moving average)
            smoothed = []
            for i in range(len(vals)):
                window = vals[max(0, i-2):i+3]
                smoothed.append(sum(window) / len(window))

            # Raw as light scatter
            fig.add_trace(go.Scatter(
                x=list(range(len(vals))),
                y=vals,
                mode="markers",
                marker=dict(size=2, color=color, opacity=0.3),
                showlegend=False,
                hovertemplate=f"Ch.%{{customdata}}: {dim}=%{{y}}<extra></extra>",
                customdata=[str(ch) for ch in valid],
            ), row=row_idx, col=col_idx)

            # Smoothed as line
            fig.add_trace(go.Scatter(
                x=list(range(len(smoothed))),
                y=smoothed,
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
            ), row=row_idx, col=col_idx)

    fig.update_layout(
        title="Dimension-by-Dimension: Two Ways to Read Rayuela",
        width=1100,
        height=250 * n_dims,
        plot_bgcolor="white",
        margin=dict(t=80),
    )
    fig.update_yaxes(range=[0.5, 10.5], dtick=3)

    return fig


# ---------------------------------------------------------------------------
# 8. Dual Heatmap: Linear vs Hopscotch reading order
# ---------------------------------------------------------------------------

def fig_dual_heatmap(scores, meta_by_num):
    """
    Side-by-side heatmaps showing the same 20-dimensional data reordered
    by the two reading paths. The visual contrast is immediate:
    linear reading shows smooth gradients; hopscotch shows rapid alternation.
    """
    # Select dimensions that are most visually informative
    display_dims = [
        "emotional_intensity", "tension_and_anxiety", "humor_and_irony",
        "existential_questioning", "love_and_desire", "death_and_mortality",
        "oliveira_centrality", "la_maga_presence", "interiority",
        "dialogue_density", "metafiction", "spatial_grounding",
    ]
    display_labels = [DIM_LABELS.get(d, d) for d in display_dims]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Linear Reading (Ch. 1→56)",
            "Hopscotch Reading (Tablero)",
        ),
        horizontal_spacing=0.08,
    )

    for col, (path, path_name) in enumerate([
        (LINEAR_ORDER, "Linear"),
        (TABLERO, "Hopscotch"),
    ], start=1):
        valid = [ch for ch in path if ch in scores]
        matrix = np.array([
            [scores[ch][d] for d in display_dims]
            for ch in valid
        ])

        # Y-axis: reading step with chapter number
        y_labels = [f"→ Ch.{ch}" for ch in valid]

        # Section color for each step
        sec_indicators = []
        for ch in valid:
            sec = meta_by_num.get(ch, {}).get("section", "")
            if "allá" in sec:
                sec_indicators.append("🔵")
            elif "acá" in sec:
                sec_indicators.append("🟠")
            else:
                sec_indicators.append("🟣")

        y_labels_with_sec = [f"{sec} {label}" for sec, label in zip(sec_indicators, y_labels)]

        hover = []
        for i, ch in enumerate(valid):
            row = []
            for j, d in enumerate(display_dims):
                row.append(f"Step {i+1}: Ch.{ch} — {display_labels[j]}: {scores[ch][d]}")
            hover.append(row)

        fig.add_trace(go.Heatmap(
            z=matrix,
            x=display_labels,
            y=y_labels,
            colorscale="RdYlBu_r",
            zmin=1, zmax=10,
            text=hover,
            hoverinfo="text",
            showscale=(col == 2),
            colorbar=dict(title="Score", x=1.02) if col == 2 else None,
        ), row=1, col=col)

    max_steps = max(
        len([ch for ch in LINEAR_ORDER if ch in scores]),
        len([ch for ch in TABLERO if ch in scores]),
    )

    fig.update_layout(
        title=dict(
            text=(
                "The Same Novel, Two Reading Orders: Narrative DNA Heatmaps<br>"
                "<sub>Each row is a reading step; columns are the 20 semantic dimensions. "
                "Linear reading shows smooth gradients; hopscotch creates deliberate discontinuity.</sub>"
            ),
            font=dict(size=14),
        ),
        width=1200,
        height=max(700, max_steps * 6 + 120),
        margin=dict(l=80, r=80, t=100, b=40),
        xaxis=dict(tickangle=-45, tickfont=dict(size=8)),
        xaxis2=dict(tickangle=-45, tickfont=dict(size=8)),
        yaxis=dict(autorange="reversed", tickfont=dict(size=6), dtick=5),
        yaxis2=dict(autorange="reversed", tickfont=dict(size=6), dtick=5),
    )

    return fig


# ---------------------------------------------------------------------------
# 9. Permutation Test: Smoothness Significance
# ---------------------------------------------------------------------------

def fig_permutation_test(scores, emb_a, chapters_meta, n_perms=5000):
    """
    Visualize the permutation test results: how smooth are the linear and
    hopscotch paths compared to random orderings?

    This is the "hero" figure for the article — it directly shows that
    the linear order is highly designed while the hopscotch is deliberately
    disruptive.
    """
    ch_to_idx = {ch["number"]: i for i, ch in enumerate(chapters_meta)}
    rng = np.random.default_rng(42)

    def path_dists(path, space):
        dists = []
        for a, b in pairwise(path):
            if space == "A":
                if a in ch_to_idx and b in ch_to_idx:
                    va, vb = emb_a[ch_to_idx[a]], emb_a[ch_to_idx[b]]
                    cos = np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb))
                    dists.append(1 - cos)
            elif space == "B":
                if a in scores and b in scores:
                    va = np.array([scores[a][d] for d in DIMS_ORDERED])
                    vb = np.array([scores[b][d] for d in DIMS_ORDERED])
                    dists.append(np.linalg.norm(va - vb))
        return np.array(dists) if dists else np.array([0.0])

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Scale A: Texture Space",
            "Scale B: Narrative Space",
        ),
        horizontal_spacing=0.12,
    )

    all_ch_a = [ch["number"] for ch in chapters_meta]
    scored_ch = list(scores.keys())

    for col, (space, pool, label) in enumerate([
        ("A", all_ch_a, "Cosine Distance"),
        ("B", scored_ch, "Euclidean Distance"),
    ], start=1):
        # Actual path means
        lin_mean = path_dists(LINEAR_ORDER, space).mean()
        hop_mean = path_dists(TABLERO, space).mean()

        # Random permutation distribution
        rand_means = []
        for _ in range(n_perms):
            perm = list(rng.permutation(pool)[:56])  # Same length as linear
            d = path_dists(perm, space)
            rand_means.append(d.mean())
        rand_means = np.array(rand_means)

        # Z-scores
        z_lin = (lin_mean - rand_means.mean()) / rand_means.std()
        z_hop = (hop_mean - rand_means.mean()) / rand_means.std()

        # Histogram of random
        fig.add_trace(go.Histogram(
            x=rand_means,
            nbinsx=50,
            marker_color="rgba(158, 158, 158, 0.5)",
            name="Random orderings" if col == 1 else None,
            showlegend=(col == 1),
            legendgroup="random",
        ), row=1, col=col)

        # Linear marker
        fig.add_vline(
            x=lin_mean, row=1, col=col,
            line=dict(color="#4CAF50", width=3),
        )
        fig.add_annotation(
            x=lin_mean,
            y=0.92,
            yref="y domain" if col == 1 else "y2 domain",
            text=f"<b>Linear</b><br>z = {z_lin:.1f}σ",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#4CAF50",
            font=dict(color="#4CAF50", size=11),
            bgcolor="white",
            bordercolor="#4CAF50",
            borderwidth=1,
            xref="x" if col == 1 else "x2",
        )

        # Hopscotch marker
        fig.add_vline(
            x=hop_mean, row=1, col=col,
            line=dict(color="#F44336", width=3),
        )
        fig.add_annotation(
            x=hop_mean,
            y=0.75,
            yref="y domain" if col == 1 else "y2 domain",
            text=f"<b>Hopscotch</b><br>z = {z_hop:+.1f}σ",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#F44336",
            font=dict(color="#F44336", size=11),
            bgcolor="white",
            bordercolor="#F44336",
            borderwidth=1,
            xref="x" if col == 1 else "x2",
        )

        fig.update_xaxes(title_text=f"Mean {label}", row=1, col=col)

    fig.update_layout(
        title=dict(
            text=(
                "Was the Reading Order Designed? Permutation Test (5,000 random orderings)<br>"
                "<sub>The linear path is far smoother than chance; the hopscotch path is indistinguishable from random</sub>"
            ),
            font=dict(size=14),
        ),
        width=1100,
        height=500,
        plot_bgcolor="white",
        yaxis=dict(title="Count"),
    )

    return fig


# ---------------------------------------------------------------------------
# 9. Section Weaving: How the Hopscotch Path Visits Sections
# ---------------------------------------------------------------------------

def fig_section_weaving(meta_by_num):
    """
    Visualize which section each step of the hopscotch path visits,
    showing the interleaving pattern of Paris / Buenos Aires / Expendable
    chapters.
    """
    sec_to_num = {
        "Del lado de allá": 0,
        "Del lado de acá": 1,
        "De otros lados (Capítulos prescindibles)": 2,
    }
    sec_colors_list = ["#2196F3", "#FF9800", "#9C27B0"]
    sec_labels = ["Allá (Paris)", "Acá (Buenos Aires)", "Otros lados"]

    # Linear path
    lin_secs = []
    lin_colors = []
    lin_ch = []
    for ch in LINEAR_ORDER:
        if ch in meta_by_num:
            sec = meta_by_num[ch]["section"]
            lin_secs.append(sec_to_num.get(sec, 2))
            lin_colors.append(sec_colors_list[sec_to_num.get(sec, 2)])
            lin_ch.append(ch)

    # Hopscotch path
    hop_secs = []
    hop_colors = []
    hop_ch = []
    for ch in TABLERO:
        if ch in meta_by_num:
            sec = meta_by_num[ch]["section"]
            hop_secs.append(sec_to_num.get(sec, 2))
            hop_colors.append(sec_colors_list[sec_to_num.get(sec, 2)])
            hop_ch.append(ch)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            "Linear Reading (Ch. 1→56): Section Sequence",
            "Hopscotch Reading (Tablero): Section Sequence",
        ),
        vertical_spacing=0.15,
    )

    # Linear
    fig.add_trace(go.Bar(
        x=list(range(len(lin_secs))),
        y=[1] * len(lin_secs),
        marker_color=lin_colors,
        hovertemplate="Step %{x}: Ch.%{customdata}<extra></extra>",
        customdata=[str(ch) for ch in lin_ch],
        showlegend=False,
    ), row=1, col=1)

    # Hopscotch
    fig.add_trace(go.Bar(
        x=list(range(len(hop_secs))),
        y=[1] * len(hop_secs),
        marker_color=hop_colors,
        hovertemplate="Step %{x}: Ch.%{customdata}<extra></extra>",
        customdata=[str(ch) for ch in hop_ch],
        showlegend=False,
    ), row=2, col=1)

    # Legend entries (invisible scatter traces)
    for i, label in enumerate(sec_labels):
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers",
            marker=dict(size=12, color=sec_colors_list[i]),
            name=label,
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Section Weaving: The Hopscotch Path Deliberately Interleaves All Three Parts<br>"
                "<sub>Linear reading follows the novel's natural section boundaries; "
                "hopscotch weaves Paris, Buenos Aires, and expendable chapters together</sub>"
            ),
            font=dict(size=14),
        ),
        width=1100,
        height=450,
        plot_bgcolor="white",
        bargap=0,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, font=dict(size=11)),
    )
    fig.update_xaxes(title_text="Reading step")
    fig.update_yaxes(visible=False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Article-ready visualizations")
    parser.add_argument("--only", choices=[
        "heatmap", "journey", "smoothness", "radar", "correlation",
        "sections", "dual", "permutation", "weaving", "dual_heatmap",
    ], help="Generate only a specific figure")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chapters_meta, dna_chapters, dims, scores, meta_by_num, emb_a = load_all_data()
    print(f"Loaded: {len(scores)} chapters with narrative DNA scores")

    def save(fig, name):
        if fig is None:
            print(f"  {name}: skipped (insufficient data)")
            return
        path = OUTPUT_DIR / f"article_{name}.html"
        fig.write_html(path)
        print(f"  Saved: {path}")

    targets = {
        "heatmap": lambda: fig_heatmap(dna_chapters, dims, scores, meta_by_num),
        "journey": lambda: fig_emotional_journeys(scores, meta_by_num),
        "smoothness": lambda: fig_trajectory_smoothness(scores, emb_a, chapters_meta),
        "radar": lambda: fig_radar_fingerprints(scores),
        "correlation": lambda: fig_correlation_matrix(scores),
        "sections": lambda: fig_section_distributions(scores, meta_by_num),
        "dual": lambda: fig_dual_journey_detail(scores),
        "dual_heatmap": lambda: fig_dual_heatmap(scores, meta_by_num),
        "permutation": lambda: fig_permutation_test(scores, emb_a, chapters_meta),
        "weaving": lambda: fig_section_weaving(meta_by_num),
    }

    if args.only:
        print(f"Generating: {args.only}")
        save(targets[args.only](), args.only)
    else:
        print("Generating all article figures...")
        for name, builder in targets.items():
            print(f"\n  [{name}]")
            save(builder(), name)

    print("\nDone!")


if __name__ == "__main__":
    main()
