#!/usr/bin/env python3
"""
3D UMAP Exploration: Scale A and Scale B with dimension reduction.

Generates interactive 3D Plotly visualizations to compare how
different dimension-reduction strategies affect the projected structure.

Experiments:
  1. Scale A (1024-dim) → 3D UMAP
  2. Scale B full (19-dim) → 3D UMAP
  3. Scale B top-8 variance (8-dim) → 3D UMAP
  4. Scale B PCA-5 (19-dim → 5 PCs) → 3D UMAP
  5. Scale B PCA-8 (19-dim → 8 PCs) → 3D UMAP
  6. Scale B de-correlated (drop redundant dims) → 3D UMAP

Usage (inside Docker container):
    python src/visualization_3d.py

    # Specific experiment:
    python src/visualization_3d.py --only scale_a

Output: outputs/figures/3d_*.html
"""

import argparse
import json

import numpy as np
import plotly.graph_objects as go
import umap

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

def load_data():
    """Load all necessary data."""
    with open(PROJECT_ROOT / "data" / "rayuela_raw.json") as f:
        raw = json.load(f)
    chapters_meta = raw["chapters"]

    emb_a = np.load(PROJECT_ROOT / "outputs" / "embeddings" / "chapter_embeddings.npy")

    with open(PROJECT_ROOT / "outputs" / "semantic" / "narrative_dna.json") as f:
        dna = json.load(f)
    from project_config import DIMS_EXCLUDED
    dims = [d for d in dna["dimensions"] if d not in DIMS_EXCLUDED]
    dna_chapters = dna["chapters"]
    scores = {ch["chapter"]: ch["scores"] for ch in dna_chapters}
    matrix_b = np.array([[scores[i + 1][d] for d in dims] for i in range(155)])

    return chapters_meta, emb_a, dims, scores, matrix_b


# ---------------------------------------------------------------------------
# Dimension reduction strategies for Scale B
# ---------------------------------------------------------------------------

def reduce_top_variance(matrix: np.ndarray, dims: list[str], k: int = 8):
    """Keep only the top-k dimensions by variance."""
    variances = matrix.var(axis=0)
    top_idx = np.argsort(-variances)[:k]
    top_idx.sort()  # Keep original order
    selected_dims = [dims[i] for i in top_idx]
    return matrix[:, top_idx], selected_dims


def reduce_pca(matrix: np.ndarray, k: int = 5):
    """PCA to k components."""
    centered = matrix - matrix.mean(axis=0)
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ Vt[:k].T
    explained = (s[:k] ** 2).sum() / (s ** 2).sum()
    return projected, explained


def reduce_decorrelated(matrix: np.ndarray, dims: list[str], threshold: float = 0.75):
    """
    Remove redundant dimensions: for each pair with |r| > threshold,
    drop the one with lower variance.
    """
    corr = np.corrcoef(matrix.T)
    variances = matrix.var(axis=0)
    drop = set()

    for i in range(len(dims)):
        if i in drop:
            continue
        for j in range(i + 1, len(dims)):
            if j in drop:
                continue
            if abs(corr[i, j]) > threshold:
                # Drop the lower-variance dimension
                victim = j if variances[j] < variances[i] else i
                drop.add(victim)

    keep = sorted(set(range(len(dims))) - drop)
    return matrix[:, keep], [dims[i] for i in keep]


# ---------------------------------------------------------------------------
# 3D UMAP projection
# ---------------------------------------------------------------------------

def run_umap_3d(
    vectors: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    metric: DistanceMetric = DistanceMetric.COSINE,
    random_state: int = 42,
) -> np.ndarray:
    """Project to 3D using UMAP."""
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(n_neighbors, len(vectors) - 1),
        min_dist=min_dist,
        metric=metric.value,
        random_state=random_state,
    )
    return reducer.fit_transform(vectors)


# ---------------------------------------------------------------------------
# 3D Plotting
# ---------------------------------------------------------------------------

def make_3d_figure(
    coords: np.ndarray,
    chapters_meta: list[dict],
    title: str,
    subtitle: str = "",
    show_trajectories: bool = True,
) -> go.Figure:
    """Create an interactive 3D scatter plot with optional trajectory overlays."""
    ch_to_idx = {ch["number"]: i for i, ch in enumerate(chapters_meta)}
    fig = go.Figure()

    # Chapter points by section
    for section, color in SECTION_COLORS.items():
        mask = [
            i for i, ch in enumerate(chapters_meta)
            if ch["section"] == section and i < len(coords)
        ]
        if not mask:
            continue

        fig.add_trace(go.Scatter3d(
            x=coords[mask, 0],
            y=coords[mask, 1],
            z=coords[mask, 2],
            mode="markers+text",
            marker=dict(
                size=4,
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            text=[str(chapters_meta[i]["number"]) for i in mask],
            textposition="top center",
            textfont=dict(size=7),
            name=SECTION_SHORT.get(section, section),
            hovertemplate=(
                "<b>Ch. %{text}</b><br>"
                f"Section: {SECTION_SHORT.get(section, section)}<br>"
                "(%{x:.2f}, %{y:.2f}, %{z:.2f})"
                "<extra></extra>"
            ),
        ))

    # Trajectory overlays
    if show_trajectories:
        for path, path_name, color, width, dash in [
            (LINEAR_ORDER, "Linear (1→56)", "rgba(76,175,80,0.5)", 3, "solid"),
            (TABLERO, "Hopscotch (Tablero)", "rgba(244,67,54,0.3)", 2, "dash"),
        ]:
            xs, ys, zs = [], [], []
            for ch_num in path:
                if ch_num in ch_to_idx and ch_to_idx[ch_num] < len(coords):
                    idx = ch_to_idx[ch_num]
                    xs.append(coords[idx, 0])
                    ys.append(coords[idx, 1])
                    zs.append(coords[idx, 2])
                else:
                    xs.append(None)
                    ys.append(None)
                    zs.append(None)

            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=path_name,
                hoverinfo="skip",
            ))

    full_title = title
    if subtitle:
        full_title += f"<br><sub>{subtitle}</sub>"

    fig.update_layout(
        title=dict(text=full_title, font=dict(size=14)),
        scene=dict(
            xaxis=dict(title="UMAP 1", showgrid=True, gridcolor="#eee"),
            yaxis=dict(title="UMAP 2", showgrid=True, gridcolor="#eee"),
            zaxis=dict(title="UMAP 3", showgrid=True, gridcolor="#eee"),
            bgcolor="white",
        ),
        width=1000,
        height=800,
        legend=dict(x=0.01, y=0.99, font=dict(size=10)),
        margin=dict(l=10, r=10, t=80, b=10),
    )

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="3D UMAP exploration")
    parser.add_argument("--only", choices=[
        "scale_a", "scale_b_full", "scale_b_top8",
        "scale_b_pca5", "scale_b_pca8", "scale_b_decorr",
    ], help="Run only one experiment")
    parser.add_argument("--n-neighbors", type=int, default=UMAP_N_NEIGHBORS)
    parser.add_argument("--min-dist", type=float, default=UMAP_MIN_DIST)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    chapters_meta, emb_a, dims, scores, matrix_b = load_data()
    print(f"Loaded: {len(chapters_meta)} chapters, Scale A={emb_a.shape}, Scale B={matrix_b.shape}")

    experiments = {}

    # --- Scale A: 1024-dim → 3D ---
    def exp_scale_a():
        print("\n=== Scale A: 1024-dim → 3D UMAP (cosine) ===")
        coords = run_umap_3d(emb_a, args.n_neighbors, args.min_dist, DistanceMetric.COSINE)
        fig = make_3d_figure(
            coords, chapters_meta,
            "Scale A: Texture Embeddings (1024-dim → 3D)",
            "Cosine metric · Each point is a chapter colored by section",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_a.html")
        print("  Saved: 3d_scale_a.html")
        return coords

    # --- Scale B full: 19-dim → 3D ---
    def exp_scale_b_full():
        print("\n=== Scale B Full: 19-dim → 3D UMAP (euclidean) ===")
        coords = run_umap_3d(matrix_b, args.n_neighbors, args.min_dist, DistanceMetric.EUCLIDEAN)
        fig = make_3d_figure(
            coords, chapters_meta,
            "Scale B Full: Narrative DNA (19-dim → 3D)",
            "Euclidean metric · All 20 dimensions",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_b_full.html")
        print("  Saved: 3d_scale_b_full.html")
        return coords

    # --- Scale B top-8 variance ---
    def exp_scale_b_top8():
        reduced, sel_dims = reduce_top_variance(matrix_b, dims, k=8)
        print(f"\n=== Scale B Top-8 Variance: {len(sel_dims)}-dim → 3D UMAP ===")
        print(f"  Selected: {sel_dims}")
        coords = run_umap_3d(reduced, args.n_neighbors, args.min_dist, DistanceMetric.EUCLIDEAN)
        fig = make_3d_figure(
            coords, chapters_meta,
            "Scale B: Top-8 Variance Dimensions → 3D",
            f"Kept: {', '.join(d.replace('_',' ').title() for d in sel_dims)}",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_b_top8var.html")
        print("  Saved: 3d_scale_b_top8var.html")
        return coords

    # --- Scale B PCA-5 ---
    def exp_scale_b_pca5():
        projected, explained = reduce_pca(matrix_b, k=5)
        print("\n=== Scale B PCA-5: 19-dim → 5 PCs → 3D UMAP ===")
        print(f"  Variance explained: {explained*100:.1f}%")
        coords = run_umap_3d(projected, args.n_neighbors, args.min_dist, DistanceMetric.EUCLIDEAN)
        fig = make_3d_figure(
            coords, chapters_meta,
            "Scale B: PCA-5 → 3D UMAP",
            f"5 principal components explaining {explained*100:.1f}% of variance",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_b_pca5.html")
        print("  Saved: 3d_scale_b_pca5.html")
        return coords

    # --- Scale B PCA-8 ---
    def exp_scale_b_pca8():
        projected, explained = reduce_pca(matrix_b, k=8)
        print("\n=== Scale B PCA-8: 19-dim → 8 PCs → 3D UMAP ===")
        print(f"  Variance explained: {explained*100:.1f}%")
        coords = run_umap_3d(projected, args.n_neighbors, args.min_dist, DistanceMetric.EUCLIDEAN)
        fig = make_3d_figure(
            coords, chapters_meta,
            "Scale B: PCA-8 → 3D UMAP",
            f"8 principal components explaining {explained*100:.1f}% of variance",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_b_pca8.html")
        print("  Saved: 3d_scale_b_pca8.html")
        return coords

    # --- Scale B de-correlated ---
    def exp_scale_b_decorr():
        reduced, sel_dims = reduce_decorrelated(matrix_b, dims, threshold=0.70)
        print(f"\n=== Scale B De-correlated: {len(sel_dims)}-dim → 3D UMAP ===")
        print(f"  Kept (|r| < 0.70 between all pairs): {sel_dims}")
        dropped = [d for d in dims if d not in sel_dims]
        print(f"  Dropped: {dropped}")
        coords = run_umap_3d(reduced, args.n_neighbors, args.min_dist, DistanceMetric.EUCLIDEAN)
        fig = make_3d_figure(
            coords, chapters_meta,
            f"Scale B: De-correlated ({len(sel_dims)}-dim → 3D)",
            f"Removed redundant dims (|r| > 0.70) · Kept {len(sel_dims)}/20",
        )
        fig.write_html(OUTPUT_DIR / "3d_scale_b_decorr.html")
        print("  Saved: 3d_scale_b_decorr.html")
        return coords

    experiments = {
        "scale_a": exp_scale_a,
        "scale_b_full": exp_scale_b_full,
        "scale_b_top8": exp_scale_b_top8,
        "scale_b_pca5": exp_scale_b_pca5,
        "scale_b_pca8": exp_scale_b_pca8,
        "scale_b_decorr": exp_scale_b_decorr,
    }

    if args.only:
        experiments[args.only]()
    else:
        print("Running all 3D experiments...")
        for _name, fn in experiments.items():
            fn()

    print("\nDone! Open the 3d_*.html files for interactive 3D exploration.")


if __name__ == "__main__":
    main()
