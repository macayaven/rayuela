#!/usr/bin/env python3
"""Convert the root article markdown files to Medium-friendly HTML."""

import re
from pathlib import Path

import markdown

STYLE = """
<style>
  body {
    font-family: Georgia, 'Times New Roman', serif;
    max-width: 740px;
    margin: 2rem auto;
    padding: 0 1rem;
    line-height: 1.8;
    color: #292929;
    font-size: 18px;
  }
  h1 { font-size: 2.2rem; margin-bottom: 0.3rem; line-height: 1.2; }
  h2 { font-size: 1.5rem; margin-top: 2.5rem; margin-bottom: 1rem; }
  p { margin-bottom: 1.2rem; }
  em { font-style: italic; }
  strong { font-weight: 700; }
  blockquote {
    border-left: 3px solid #ccc;
    padding-left: 1rem;
    color: #555;
    margin: 1.5rem 0;
  }
  table {
    border-collapse: collapse;
    margin: 1.5rem 0;
    width: 100%;
    font-size: 0.95rem;
  }
  th, td {
    border: 1px solid #ddd;
    padding: 0.5rem 0.75rem;
    text-align: left;
  }
  th { background: #f5f5f5; font-weight: 600; }
  hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1.5rem 0;
  }
  .figure-caption {
    font-size: 0.9rem;
    color: #666;
    font-style: italic;
    margin-top: -1rem;
    margin-bottom: 2rem;
  }
  ul, ol { margin-bottom: 1.2rem; padding-left: 1.5rem; }
  li { margin-bottom: 0.4rem; }
  a { color: #1a8917; }
  code { background: #f5f5f5; padding: 0.15rem 0.3rem; border-radius: 3px; font-size: 0.9em; }
  .byline { color: #757575; font-size: 1rem; margin-bottom: 2rem; }
  .attribution {
    background: #f9f9f9;
    border: 1px solid #eee;
    padding: 1rem 1.2rem;
    border-radius: 4px;
    font-size: 0.9rem;
    color: #555;
    margin-top: 2rem;
  }
</style>
"""

IMAGE_BASE_URL = "https://raw.githubusercontent.com/macayaven/rayuela/main/article_images/"


def convert(md_path: Path, html_path: Path) -> None:
    """Render one article markdown file to standalone HTML."""
    text = md_path.read_text(encoding="utf-8")

    # Root article markdown uses bare PNG names; rewrite them to stable
    # raw GitHub URLs so the exported HTML works outside the repo checkout.
    text = re.sub(
        r"\]\((p[12]_[^)]+\.(?:png|jpe?g|gif|webp))\)",
        lambda match: f"]({IMAGE_BASE_URL}{match.group(1)})",
        text,
    )

    # Convert image captions (lines starting with * after images) to styled divs
    lines = text.split("\n")
    processed = []
    for line in lines:
        if (
            line.startswith("*Figure")
            or line.startswith("*This is Part")
            or line.startswith("*All interactive")
        ):
            processed.append(f'<p class="figure-caption">{line.strip("*")}</p>')
        elif line.startswith("*This article is the result"):
            processed.append(f'<div class="attribution">{line.strip("*")}</div>')
        else:
            processed.append(line)
    text = "\n".join(processed)

    html_body = markdown.markdown(
        text,
        extensions=["tables", "smarty"],
        output_format="html5",
    )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{md_path.stem}</title>
  {STYLE}
</head>
<body>
{html_body}
</body>
</html>
"""
    html_path.write_text(html_doc, encoding="utf-8")
    print(f"  {md_path.name} -> {html_path.name}")


def main(base: Path | None = None) -> None:
    """Regenerate the root HTML article exports from the current markdown sources."""
    if base is None:
        base = Path(__file__).resolve().parent.parent

    article_pairs = [
        ("ARTICLE_PART1_MEDIUM.md", "ARTICLE_PART1_MEDIUM.html"),
        ("ARTICLE_PART2_MEDIUM.md", "ARTICLE_PART2_MEDIUM.html"),
    ]

    for md_name, html_name in article_pairs:
        md_file = base / md_name
        html_file = base / html_name
        if md_file.exists():
            convert(md_file, html_file)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
