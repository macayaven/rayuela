#!/usr/bin/env python3
"""
Prepare visualization files for GitHub Pages deployment.

Strips the embedded Plotly.js library from each HTML file and replaces
it with a CDN reference, reducing file sizes from ~4.7 MB to ~100-200 KB.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "outputs" / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
ARTICLE_SOURCES = [
    PROJECT_ROOT / "ARTICLE_PART1_MEDIUM.md",
    PROJECT_ROOT / "ARTICLE_PART2_MEDIUM.md",
]
PAGES_BASE_URL = "https://carloscrespomacaya.com/rayuela/"

# Plotly CDN URL (pinned version to match the embedded one)
PLOTLY_CDN = '<script src="https://cdn.plot.ly/plotly-3.3.1.min.js"></script>'

# Files to include (article figures + key visualizations)
INCLUDE_FILES = [
    # Article figures (referenced in Part 1)
    "article_permutation.html",  # Figure 3 — the hero figure
    "article_smoothness.html",  # Figure 3 in old draft
    "article_weaving.html",  # Figure 4
    "article_journey.html",  # Figure 5
    "article_heatmap.html",  # Full narrative DNA
    "article_radar.html",  # Chapter fingerprints
    "article_dual_heatmap.html",  # Two reading orders compared
    "article_dual.html",  # Dimension-by-dimension
    "article_sections.html",  # Section profiles
    "article_correlation.html",  # Dimension correlations
    # UMAP projections
    "umap_comparison.html",  # Figure 2
    "umap_scale_a.html",
    "umap_scale_b.html",
    # 3D explorations
    "3d_scale_a.html",  # Figure 1
    "3d_scale_b_full.html",
    "3d_scale_b_top8var.html",
    "3d_scale_b_pca5.html",
    "3d_scale_b_pca8.html",
    "3d_scale_b_decorr.html",
]


def extract_published_targets(markdown: str, *, base_url: str = PAGES_BASE_URL) -> list[str]:
    """Extract published Pages HTML targets referenced from article markdown."""
    pattern = re.compile(rf"{re.escape(base_url)}([^)\s?#]+)")
    targets: list[str] = []
    for match in pattern.finditer(markdown):
        target = match.group(1)
        if target.endswith(".html") and target not in targets:
            targets.append(target)
    return targets


def validate_published_targets(
    *,
    article_paths: list[Path] = ARTICLE_SOURCES,
    docs_dir: Path = DOCS_DIR,
    include_files: list[str] = INCLUDE_FILES,
    base_url: str = PAGES_BASE_URL,
) -> tuple[list[str], list[str]]:
    """Return missing docs targets and missing publish-bundle registrations."""
    referenced_targets: list[str] = []
    for article_path in article_paths:
        markdown = article_path.read_text(encoding="utf-8")
        for target in extract_published_targets(markdown, base_url=base_url):
            if target not in referenced_targets:
                referenced_targets.append(target)

    missing_docs = [target for target in referenced_targets if not (docs_dir / target).exists()]
    missing_publish_bundle = [
        target for target in referenced_targets if target not in include_files
    ]
    return missing_docs, missing_publish_bundle


def strip_inline_plotly(html: str) -> str:
    """Replace the inline Plotly.js script with a CDN reference."""
    # Plotly embeds as: <script type="text/javascript">...plotly.js v3.3.1...</script>
    # followed by data/layout scripts
    # Strategy: find the big script block containing plotly.js and replace it

    # Match the script tag containing the Plotly library (identified by version string)
    pattern = r'<script type="text/javascript">/\*\*\s*\n\* plotly\.js v[\d.]+.*?</script>'

    replacement = PLOTLY_CDN
    result, count = re.subn(pattern, replacement, html, count=1, flags=re.DOTALL)

    if count == 0:
        # Try alternative pattern (PlotlyConfig line + library)
        pattern2 = (
            r'<script type="text/javascript">window\.PlotlyConfig.*?</script>\s*'
            r'<script type="text/javascript">/\*\*\s*\n\* plotly\.js.*?</script>'
        )
        result, count = re.subn(pattern2, PLOTLY_CDN, html, count=1, flags=re.DOTALL)

    if count == 0:
        print("  WARNING: Could not find inline Plotly.js to strip")
        return html

    return result


def create_index_page(files: list[str]) -> str:
    """Generate the index.html page for GitHub Pages."""

    # Organize files by category
    article_figs = [f for f in files if f.startswith("article_")]
    umap_figs = [f for f in files if f.startswith("umap_")]
    threed_figs = [f for f in files if f.startswith("3d_")]

    def make_card(filename: str) -> str:
        # Convert filename to readable title
        name = (
            filename.replace(".html", "")
            .replace("article_", "")
            .replace("umap_", "UMAP: ")
            .replace("3d_", "3D: ")
        )
        name = name.replace("_", " ").title()
        return f'      <a href="{filename}" class="card">{name}</a>'

    article_cards = "\n".join(make_card(f) for f in article_figs)
    umap_cards = "\n".join(make_card(f) for f in umap_figs)
    threed_cards = "\n".join(make_card(f) for f in threed_figs)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Rayuela — Interactive Visualizations</title>
  <style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
      font-family: 'Source Sans 3', 'Source Sans Pro', -apple-system, sans-serif;
      background: #fafafa;
      color: #333;
      line-height: 1.6;
      padding: 2rem;
      max-width: 900px;
      margin: 0 auto;
    }}
    h1 {{
      font-family: 'Merriweather', Georgia, serif;
      font-size: 2rem;
      margin-bottom: 0.5rem;
      color: #1a1a1a;
    }}
    .subtitle {{
      font-size: 1.1rem;
      color: #666;
      margin-bottom: 2rem;
    }}
    .intro {{
      font-size: 1rem;
      color: #555;
      margin-bottom: 2rem;
      border-left: 3px solid #2196F3;
      padding-left: 1rem;
    }}
    h2 {{
      font-size: 1.3rem;
      color: #1a1a1a;
      margin: 2rem 0 1rem;
      padding-bottom: 0.3rem;
      border-bottom: 2px solid #eee;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
      gap: 1rem;
      margin-bottom: 2rem;
    }}
    .card {{
      display: block;
      padding: 1.2rem;
      background: white;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
      text-decoration: none;
      color: #333;
      font-weight: 500;
      transition: all 0.2s;
    }}
    .card:hover {{
      border-color: #2196F3;
      box-shadow: 0 2px 8px rgba(33, 150, 243, 0.15);
      transform: translateY(-1px);
    }}
    footer {{
      margin-top: 3rem;
      padding-top: 1rem;
      border-top: 1px solid #eee;
      font-size: 0.85rem;
      color: #999;
    }}
    footer a {{ color: #666; }}
  </style>
</head>
<body>

<h1>What Does a Novel Look Like From the Inside?</h1>
<p class="subtitle">
  Interactive Visualizations for the Computational Analysis of Cortazar's <em>Rayuela</em>
</p>

<p class="intro">
  These interactive charts accompany our article on using AI to map the hidden structure of
  Julio Cortazar's <em>Rayuela</em> (1963). Each visualization is built with Plotly and is
  fully interactive: hover for details, drag to pan, scroll to zoom, and click legend items
  to toggle layers. 3D charts can be orbited by dragging.
</p>

<h2>Article Figures</h2>
<div class="grid">
{article_cards}
</div>

<h2>UMAP Projections</h2>
<div class="grid">
{umap_cards}
</div>

<h2>3D Explorations</h2>
<div class="grid">
{threed_cards}
</div>

<footer>
  Carlos Crespo Macaya &amp; Claude &middot; 2026 &middot;
  <a href="https://github.com/macayaven/rayuela">Source Code</a>
</footer>

</body>
</html>
"""


def main() -> None:
    """Copy selected Plotly exports into docs/ and rebuild the Pages index."""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    processed = []
    for filename in INCLUDE_FILES:
        src = SOURCE_DIR / filename
        if not src.exists():
            print(f"  SKIP: {filename} (not found)")
            continue

        html = src.read_text()
        original_size = len(html)

        stripped = strip_inline_plotly(html)
        new_size = len(stripped)

        dst = DOCS_DIR / filename
        dst.write_text(stripped)

        reduction = (1 - new_size / original_size) * 100
        print(f"  {filename}: {original_size:,} -> {new_size:,} ({reduction:.0f}% reduction)")
        processed.append(filename)

    # Create index page
    index = create_index_page(processed)
    (DOCS_DIR / "index.html").write_text(index)
    print(f"\n  index.html created with {len(processed)} visualizations")

    # Summary
    total_src = sum((SOURCE_DIR / f).stat().st_size for f in processed)
    total_dst = sum((DOCS_DIR / f).stat().st_size for f in processed)
    total_dst += (DOCS_DIR / "index.html").stat().st_size
    print(f"\n  Total: {total_src / 1e6:.1f} MB -> {total_dst / 1e6:.1f} MB")


if __name__ == "__main__":  # pragma: no cover
    main()
