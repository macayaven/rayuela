from __future__ import annotations

from pathlib import Path

from .conftest import load_script_module

md_to_html = load_script_module("md_to_html_test", "scripts/md_to_html.py")


def test_convert_rewrites_images_and_formats_special_blocks(tmp_path: Path) -> None:
    md_path = tmp_path / "article.md"
    html_path = tmp_path / "article.html"
    md_path.write_text(
        "\n".join(
            [
                "# Sample",
                "![A figure](p1_figure1_3d_novel.png)",
                "*Figure 1 — Caption*",
                "*This article is the result of a collaboration.*",
            ]
        ),
        encoding="utf-8",
    )

    md_to_html.convert(md_path, html_path)
    html = html_path.read_text(encoding="utf-8")

    assert "https://raw.githubusercontent.com/macayaven/rayuela/main/article_images/" in html
    assert 'class="figure-caption"' in html
    assert 'class="attribution"' in html
    assert "<title>article</title>" in html


def test_main_converts_existing_root_articles(tmp_path: Path) -> None:
    (tmp_path / "ARTICLE_PART1_MEDIUM.md").write_text("# Part 1", encoding="utf-8")
    (tmp_path / "ARTICLE_PART2_MEDIUM.md").write_text("# Part 2", encoding="utf-8")

    md_to_html.main(tmp_path)

    assert (tmp_path / "ARTICLE_PART1_MEDIUM.html").exists()
    assert (tmp_path / "ARTICLE_PART2_MEDIUM.html").exists()
