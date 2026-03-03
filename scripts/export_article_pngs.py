#!/usr/bin/env python3
"""
Export article figures as static PNG files for Medium.

Generates high-resolution PNGs for the 5 figures referenced in ARTICLE_PART1.md:
  Figure 1: 3D Novel Map (extracted from existing HTML)
  Figure 2: UMAP Comparison (extracted from existing HTML)
  Figure 3: Permutation Test (regenerated from data)
  Figure 4: Section Weaving (regenerated from data)
  Figure 5: Emotional Arcs (regenerated from data)

Usage (inside Docker container — kaleido must be installed):
    pip install kaleido && python scripts/export_article_pngs.py
"""

import json
import re
import sys
from pathlib import Path

# Add src/ to path so we can import article_figures
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import plotly.graph_objects as go
import plotly.io as pio

from article_figures import (
    load_all_data,
    fig_permutation_test,
    fig_section_weaving,
    fig_emotional_journeys,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures" / "png"
HTML_DIR = PROJECT_ROOT / "outputs" / "figures"

# PNG export settings
PNG_SCALE = 3  # 3x resolution for crisp Medium images
PNG_WIDTH = 1100
PNG_HEIGHT_DEFAULT = 600


def extract_figure_from_html(html_path: Path) -> go.Figure:
    """Extract a Plotly figure object from a saved HTML file."""
    html = html_path.read_text()

    # Plotly HTML files store figure data in a Plotly.newPlot() or Plotly.react() call
    # The data is JSON between the first [...] and {...} arguments
    # Pattern: Plotly.newPlot("div-id", data, layout, config)
    # Or in newer plotly: the full figure JSON is in a script block

    # Try to find the plotly JSON figure specification
    # Plotly stores it as: {"data": [...], "layout": {...}}
    pattern = r'Plotly\.(?:newPlot|react)\(\s*["\'][\w-]+["\']\s*,\s*(\[.*?\])\s*,\s*(\{.*?\})\s*[,\)]'
    match = re.search(pattern, html, re.DOTALL)

    if match:
        data_json = match.group(1)
        layout_json = match.group(2)
        # Clean up any trailing config object from layout
        data = json.loads(data_json)
        layout = json.loads(layout_json)
        return go.Figure(data=data, layout=layout)

    # Alternative: plotly sometimes uses window.PlotlyConfig + full figure JSON
    pattern2 = r'"data":\s*(\[.*?\])\s*,\s*"layout":\s*(\{.*?\})\s*(?:,\s*"config"|[}\)])'
    match2 = re.search(pattern2, html, re.DOTALL)

    if match2:
        data = json.loads(match2.group(1))
        layout = json.loads(match2.group(2))
        return go.Figure(data=data, layout=layout)

    raise ValueError(f"Could not extract Plotly figure from {html_path}")


def save_png(fig: go.Figure, name: str, width: int = PNG_WIDTH, height: int = PNG_HEIGHT_DEFAULT):
    """Save a Plotly figure as a high-resolution PNG."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.write_image(
        str(path),
        format="png",
        width=width,
        height=height,
        scale=PNG_SCALE,
    )
    size_kb = path.stat().st_size / 1024
    print(f"  {name}.png: {size_kb:.0f} KB ({width*PNG_SCALE}x{height*PNG_SCALE}px)")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data for regenerated figures
    print("Loading data...")
    chapters_meta, dna_chapters, dims, scores, meta_by_num, emb_a = load_all_data()
    print(f"  {len(scores)} chapters loaded\n")

    # --- Figure 1: 3D Novel Map (from HTML) ---
    print("[Figure 1] 3D Novel Map")
    try:
        fig1 = extract_figure_from_html(HTML_DIR / "3d_scale_a.html")
        # Set a good static camera angle for the 3D plot
        fig1.update_layout(
            scene_camera=dict(
                eye=dict(x=1.5, y=1.5, z=0.8),
                up=dict(x=0, y=0, z=1),
            ),
            margin=dict(l=0, r=0, t=60, b=0),
        )
        save_png(fig1, "figure1_3d_novel", width=1000, height=750)
    except Exception as e:
        print(f"  SKIP: {e}")

    # --- Figure 2: UMAP Comparison (from HTML) ---
    print("[Figure 2] UMAP Comparison")
    try:
        fig2 = extract_figure_from_html(HTML_DIR / "umap_comparison.html")
        save_png(fig2, "figure2_umap_comparison", width=1200, height=550)
    except Exception as e:
        print(f"  SKIP: {e}")

    # --- Figure 3: Permutation Test (regenerated) ---
    print("[Figure 3] Permutation Test (computing 5,000 permutations...)")
    fig3 = fig_permutation_test(scores, emb_a, chapters_meta, n_perms=5000)
    save_png(fig3, "figure3_permutation", width=1100, height=500)

    # --- Figure 4: Section Weaving (regenerated) ---
    print("[Figure 4] Section Weaving")
    fig4 = fig_section_weaving(meta_by_num)
    save_png(fig4, "figure4_weaving", width=1100, height=450)

    # --- Figure 5: Emotional Arcs (regenerated) ---
    print("[Figure 5] Emotional Arcs")
    fig5 = fig_emotional_journeys(scores, meta_by_num)
    save_png(fig5, "figure5_emotional_arcs", width=1100, height=700)

    print(f"\nAll PNGs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
