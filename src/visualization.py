#!/usr/bin/env python3
"""
Phase 4: UMAP Dimensionality Reduction & Trajectory Visualization.

Produces interactive Plotly visualizations of the two scales:
  - Scale A (texture embeddings, 1024-dim → 2D)
  - Scale B (narrative DNA, 19-dim → 2D)

Each plot shows:
  1. Chapters colored by section and expendability
  2. The linear reading path (Ch. 1→56) as a trajectory
  3. The hopscotch reading path (Tablero de Dirección) as a trajectory

Usage (inside Docker container):
    python src/visualization.py

    # Scale A only (if Scale B is still running):
    python src/visualization.py --scale-a-only

    # Custom UMAP parameters:
    python src/visualization.py --n-neighbors 25 --min-dist 0.15

Output: outputs/figures/umap_*.html (interactive Plotly)
        outputs/figures/umap_*.png (static)
"""

import argparse
import json

import numpy as np
import plotly.graph_objects as go
import umap
from plotly.subplots import make_subplots

from project_config import (
    LINEAR_ORDER,
    OUTPUT_FIGURES_DIR,
    PROJECT_ROOT,
    SECTION_COLORS,
    SECTION_SHORT,
    TABLERO,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    DistanceMetric,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR = OUTPUT_FIGURES_DIR


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_chapter_metadata() -> list[dict]:
    """Load chapter metadata from rayuela_raw.json."""
    data_path = PROJECT_ROOT / "data" / "rayuela_raw.json"
    with open(data_path) as f:
        data = json.load(f)
    return data["chapters"]


def load_scale_a() -> np.ndarray:
    """Load Scale A texture embeddings (155 × 1024)."""
    path = PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy"
    return np.load(path)


def load_scale_b() -> tuple[np.ndarray | None, list[dict] | None]:
    """
    Load Scale B narrative DNA vectors.

    Returns (vectors, chapters) or (None, None) if not available.
    Handles partial extractions (< 155 chapters).
    """
    json_path = PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json"
    if not json_path.exists():
        return None, None

    with open(json_path) as f:
        data = json.load(f)

    chapters = data.get("chapters", [])
    if not chapters:
        return None, None

    from project_config import DIMS_EXCLUDED
    dims = [d for d in data["dimensions"] if d not in DIMS_EXCLUDED]
    vectors = np.array(
        [[ch["scores"][d] for d in dims] for ch in chapters],
        dtype=np.float32,
    )
    return vectors, chapters


# ---------------------------------------------------------------------------
# UMAP projection
# ---------------------------------------------------------------------------

def run_umap(
    vectors: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: DistanceMetric = DistanceMetric.COSINE,
    random_state: int = 42,
) -> np.ndarray:
    """Project high-dimensional vectors to 2D using UMAP."""
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric.value,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def chapter_to_index(chapters_meta: list[dict]) -> dict[int, int]:
    """Map chapter number → index in the arrays."""
    return {ch["number"]: i for i, ch in enumerate(chapters_meta)}


def build_trajectory(
    reading_order: list[int],
    coords_2d: np.ndarray,
    ch_to_idx: dict[int, int],
) -> tuple[list[float], list[float]]:
    """
    Build x, y coordinate lists for a reading path trajectory.

    Skips chapters not present in the coordinate space.
    Returns (xs, ys) with None separators for breaks.
    """
    xs, ys = [], []
    for ch_num in reading_order:
        if ch_num in ch_to_idx:
            idx = ch_to_idx[ch_num]
            xs.append(coords_2d[idx, 0])
            ys.append(coords_2d[idx, 1])
        else:
            # Insert break in the trajectory
            xs.append(None)
            ys.append(None)
    return xs, ys


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_umap_figure(
    coords_2d: np.ndarray,
    chapters_meta: list[dict],
    title: str,
    ch_to_idx: dict[int, int] | None = None,
    show_trajectories: bool = True,
) -> go.Figure:
    """
    Create an interactive UMAP scatter plot with optional trajectory overlays.

    Each chapter is a point colored by section, sized by word count.
    """
    if ch_to_idx is None:
        ch_to_idx = chapter_to_index(chapters_meta)

    fig = go.Figure()

    # --- Chapter points, grouped by section ---
    for section, color in SECTION_COLORS.items():
        mask = [
            i for i, ch in enumerate(chapters_meta)
            if ch["section"] == section and i < len(coords_2d)
        ]
        if not mask:
            continue

        fig.add_trace(go.Scatter(
            x=coords_2d[mask, 0],
            y=coords_2d[mask, 1],
            mode="markers+text",
            marker=dict(
                size=[max(6, min(18, chapters_meta[i]["token_count"] / 200)) for i in mask],
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=[str(chapters_meta[i]["number"]) for i in mask],
            textposition="top center",
            textfont=dict(size=8),
            name=SECTION_SHORT.get(section, section),
            hovertemplate=(
                "<b>Ch. %{text}</b><br>"
                + f"Section: {SECTION_SHORT.get(section, section)}<br>"
                + "UMAP: (%{x:.2f}, %{y:.2f})<br>"
                + "<extra></extra>"
            ),
        ))

    # --- Trajectory overlays ---
    if show_trajectories:
        # Linear path (1→56)
        lx, ly = build_trajectory(LINEAR_ORDER, coords_2d, ch_to_idx)
        fig.add_trace(go.Scatter(
            x=lx, y=ly,
            mode="lines",
            line=dict(color="rgba(76, 175, 80, 0.3)", width=1.5, dash="dot"),
            name="Linear (1→56)",
            hoverinfo="skip",
        ))

        # Hopscotch path
        hx, hy = build_trajectory(TABLERO, coords_2d, ch_to_idx)
        fig.add_trace(go.Scatter(
            x=hx, y=hy,
            mode="lines",
            line=dict(color="rgba(244, 67, 54, 0.25)", width=1),
            name="Hopscotch (Tablero)",
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="UMAP 1", showgrid=False),
        yaxis=dict(title="UMAP 2", showgrid=False),
        plot_bgcolor="white",
        width=900,
        height=700,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        margin=dict(l=50, r=50, t=60, b=50),
    )

    return fig


def make_comparison_figure(
    coords_a: np.ndarray,
    coords_b: np.ndarray,
    meta_a: list[dict],
    meta_b_chapters: list[dict],
    all_meta: list[dict],
) -> go.Figure:
    """
    Side-by-side Scale A vs Scale B UMAP comparison.

    Scale B may have fewer chapters than Scale A, so we align by chapter number.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Scale A: Texture (1024-dim)", "Scale B: Narrative DNA (19-dim)"),
        horizontal_spacing=0.08,
    )

    ch_to_idx_a = chapter_to_index(all_meta)
    ch_to_idx_b = {ch["chapter"]: i for i, ch in enumerate(meta_b_chapters)}

    for col, (coords, ch_idx, _title_suffix) in enumerate([
        (coords_a, ch_to_idx_a, "A"),
        (coords_b, ch_to_idx_b, "B"),
    ], start=1):
        for section, color in SECTION_COLORS.items():
            # Find chapters in this section that exist in this scale
            mask = []
            labels = []
            for ch in all_meta:
                if ch["section"] == section and ch["number"] in ch_idx:
                    idx = ch_idx[ch["number"]]
                    if idx < len(coords):
                        mask.append(idx)
                        labels.append(str(ch["number"]))

            if not mask:
                continue

            fig.add_trace(
                go.Scatter(
                    x=coords[mask, 0],
                    y=coords[mask, 1],
                    mode="markers+text",
                    marker=dict(size=8, color=color, opacity=0.8),
                    text=labels,
                    textposition="top center",
                    textfont=dict(size=7),
                    name=SECTION_SHORT.get(section, section),
                    showlegend=(col == 1),  # Only show legend once
                ),
                row=1, col=col,
            )

    fig.update_layout(
        title="Scale A (Texture) vs Scale B (Narrative DNA)",
        width=1600,
        height=700,
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 4: UMAP visualization")
    parser.add_argument("--scale-a-only", action="store_true",
                        help="Only visualize Scale A (skip Scale B)")
    parser.add_argument("--n-neighbors", type=int, default=UMAP_N_NEIGHBORS,
                        help=f"UMAP n_neighbors (default: {UMAP_N_NEIGHBORS})")
    parser.add_argument("--min-dist", type=float, default=UMAP_MIN_DIST,
                        help=f"UMAP min_dist (default: {UMAP_MIN_DIST})")
    parser.add_argument("--no-trajectories", action="store_true",
                        help="Don't overlay reading paths")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chapters_meta = load_chapter_metadata()
    print(f"Loaded {len(chapters_meta)} chapters metadata")

    # --- Scale A ---
    print("\n=== Scale A: Texture Embeddings ===")
    embeddings_a = load_scale_a()
    print(f"Embeddings: {embeddings_a.shape}")

    coords_a = run_umap(
        embeddings_a,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=DistanceMetric.COSINE,
    )
    print(f"UMAP projection: {coords_a.shape}")

    fig_a = make_umap_figure(
        coords_a, chapters_meta,
        title="Scale A: Texture Embeddings (UMAP, cosine)",
        show_trajectories=not args.no_trajectories,
    )
    fig_a.write_html(OUTPUT_DIR / "umap_scale_a.html")
    print(f"Saved: {OUTPUT_DIR / 'umap_scale_a.html'}")
    try:
        fig_a.write_image(OUTPUT_DIR / "umap_scale_a.png", scale=2)
        print(f"Saved: {OUTPUT_DIR / 'umap_scale_a.png'}")
    except (ValueError, ImportError):
        print("(PNG export skipped — kaleido not installed)")

    # Save UMAP coordinates for later analysis
    np.save(OUTPUT_DIR / "umap_coords_scale_a.npy", coords_a)

    if args.scale_a_only:
        print("\n(Skipping Scale B — --scale-a-only flag)")
        return

    # --- Scale B ---
    print("\n=== Scale B: Narrative DNA ===")
    vectors_b, chapters_b = load_scale_b()
    if vectors_b is None or len(vectors_b) < 10:
        print(f"Scale B: insufficient data ({0 if vectors_b is None else len(vectors_b)} chapters)")
        print("Run visualization again when Scale B extraction completes.")
        return

    print(f"Vectors: {vectors_b.shape} ({len(chapters_b)} chapters)")

    coords_b = run_umap(
        vectors_b,
        n_neighbors=min(args.n_neighbors, len(vectors_b) - 1),
        min_dist=args.min_dist,
        metric=DistanceMetric.EUCLIDEAN,
    )
    print(f"UMAP projection: {coords_b.shape}")

    # Build chapter-to-index mapping for Scale B (may be partial)
    ch_to_idx_b = {ch["chapter"]: i for i, ch in enumerate(chapters_b)}

    fig_b = make_umap_figure(
        coords_b, chapters_meta,
        title=f"Scale B: Narrative DNA (UMAP, euclidean) — {len(chapters_b)}/155 chapters",
        ch_to_idx=ch_to_idx_b,
        show_trajectories=not args.no_trajectories,
    )
    fig_b.write_html(OUTPUT_DIR / "umap_scale_b.html")
    print(f"Saved: {OUTPUT_DIR / 'umap_scale_b.html'}")
    try:
        fig_b.write_image(OUTPUT_DIR / "umap_scale_b.png", scale=2)
        print(f"Saved: {OUTPUT_DIR / 'umap_scale_b.png'}")
    except (ValueError, ImportError):
        print("(PNG export skipped — kaleido not installed)")

    np.save(OUTPUT_DIR / "umap_coords_scale_b.npy", coords_b)

    # --- Comparison ---
    print("\n=== Comparison: Scale A vs Scale B ===")
    fig_comp = make_comparison_figure(
        coords_a, coords_b,
        chapters_meta, chapters_b, chapters_meta,
    )
    fig_comp.write_html(OUTPUT_DIR / "umap_comparison.html")
    print(f"Saved: {OUTPUT_DIR / 'umap_comparison.html'}")
    try:
        fig_comp.write_image(OUTPUT_DIR / "umap_comparison.png", scale=2)
        print(f"Saved: {OUTPUT_DIR / 'umap_comparison.png'}")
    except (ValueError, ImportError):
        print("(PNG export skipped — kaleido not installed)")

    print("\nDone! Open the .html files for interactive exploration.")


if __name__ == "__main__":
    main()
