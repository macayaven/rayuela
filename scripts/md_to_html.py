#!/usr/bin/env python3
"""Convert the root article markdown files to Medium-friendly HTML."""

import argparse
import os
import re
import subprocess
from collections.abc import Sequence
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

ARTICLE_IMAGES_DIR = "article_images"
IMAGE_BASE_URL_ENV = "RAYUELA_IMAGE_BASE_URL"
GITHUB_REMOTE_RE = re.compile(
    r"(?:https://github\.com/|git@github\.com:)(?P<repo>[^/]+/[^/.]+?)(?:\.git)?$"
)
MARKDOWN_EXTENSIONS = ["tables", "smarty"]
INLINE_MARKDOWN_EXTENSIONS = ["smarty"]


def normalize_image_base_url(base_url: str) -> str:
    """Ensure the configured image base URL ends with a slash."""
    return base_url if base_url.endswith("/") else f"{base_url}/"


def parse_github_repo(remote_url: str) -> str | None:
    """Extract the owner/repo slug from a GitHub remote URL."""
    match = GITHUB_REMOTE_RE.fullmatch(remote_url.strip())
    if match is None:
        return None
    return match.group("repo")


def run_git(args: Sequence[str], cwd: Path) -> str | None:
    """Run one git command and return stdout when it succeeds."""
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        return None
    output = result.stdout.strip()
    return output or None


def resolve_image_base_url(base: Path, explicit: str | None = None) -> str:
    """Pick the image base URL from CLI/env/git metadata, in that order."""
    if explicit:
        return normalize_image_base_url(explicit)

    env_base_url = os.getenv(IMAGE_BASE_URL_ENV)
    if env_base_url:
        return normalize_image_base_url(env_base_url)

    remote_url = run_git(["config", "--get", "remote.origin.url"], base)
    commit_sha = run_git(["rev-parse", "HEAD"], base)
    repo_slug = parse_github_repo(remote_url) if remote_url else None
    if repo_slug and commit_sha:
        return f"https://raw.githubusercontent.com/{repo_slug}/{commit_sha}/{ARTICLE_IMAGES_DIR}/"

    return f"{ARTICLE_IMAGES_DIR}/"


def render_special_block(text: str, css_class: str) -> str:
    """Render one special markdown line to styled HTML without losing links."""
    content = text.strip().strip("*").strip()
    rendered = markdown.markdown(
        content,
        extensions=INLINE_MARKDOWN_EXTENSIONS,
        output_format="html5",
    )
    if css_class == "figure-caption":
        return rendered.replace("<p>", f'<p class="{css_class}">', 1)
    return f'<div class="{css_class}">{rendered}</div>'


def preprocess_special_blocks(text: str) -> str:
    """Convert caption and attribution marker lines to styled HTML blocks."""
    processed: list[str] = []
    for line in text.splitlines():
        if (
            line.startswith("*Figure")
            or line.startswith("*This is Part")
            or line.startswith("*All interactive")
        ):
            processed.append(render_special_block(line, "figure-caption"))
        elif line.startswith("*This article is the result"):
            processed.append(render_special_block(line, "attribution"))
        else:
            processed.append(line)
    return "\n".join(processed)


def convert(md_path: Path, html_path: Path, image_base_url: str) -> None:
    """Render one article markdown file to standalone HTML."""
    text = md_path.read_text(encoding="utf-8")
    normalized_image_base_url = normalize_image_base_url(image_base_url)

    # Root article markdown uses bare PNG names; rewrite them to stable
    # image URLs so the exported HTML works outside the repo checkout.
    text = re.sub(
        r"\]\((p[12]_[^)]+\.(?:png|jpe?g|gif|webp))\)",
        lambda match: f"]({normalized_image_base_url}{match.group(1)})",
        text,
    )
    text = preprocess_special_blocks(text)

    html_body = markdown.markdown(
        text,
        extensions=MARKDOWN_EXTENSIONS,
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for export configuration."""
    parser = argparse.ArgumentParser(
        description="Convert the root article markdown files to Medium-friendly HTML."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Repository root containing the ARTICLE_PART*_MEDIUM.md files.",
    )
    parser.add_argument(
        "--image-base-url",
        default=None,
        help=(
            f"Base URL for article images. Overrides the {IMAGE_BASE_URL_ENV} environment variable."
        ),
    )
    return parser.parse_args(argv)


def main(base: Path | None = None, image_base_url: str | None = None) -> None:
    """Regenerate the root HTML article exports from the current markdown sources."""
    if base is None:
        base = Path(__file__).resolve().parent.parent

    resolved_image_base_url = resolve_image_base_url(base, image_base_url)
    article_pairs = [
        ("ARTICLE_PART1_MEDIUM.md", "ARTICLE_PART1_MEDIUM.html"),
        ("ARTICLE_PART2_MEDIUM.md", "ARTICLE_PART2_MEDIUM.html"),
    ]

    for md_name, html_name in article_pairs:
        md_file = base / md_name
        html_file = base / html_name
        if md_file.exists():
            convert(md_file, html_file, resolved_image_base_url)
    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    args = parse_args()
    main(args.base_dir, args.image_base_url)
