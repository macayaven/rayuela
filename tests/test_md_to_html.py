from __future__ import annotations

from pathlib import Path

import pytest

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
                "*Figure 1 — Caption with an [Interactive version](https://example.com/view)*",
                "*This article is the result of a [collaboration](https://example.com/about).*",
            ]
        ),
        encoding="utf-8",
    )

    md_to_html.convert(
        md_path,
        html_path,
        "https://cdn.example.test/article_images",
    )
    html = html_path.read_text(encoding="utf-8")

    assert "https://cdn.example.test/article_images/p1_figure1_3d_novel.png" in html
    assert 'class="figure-caption"' in html
    assert 'class="attribution"' in html
    assert 'href="https://example.com/view"' in html
    assert 'href="https://example.com/about"' in html
    assert "[Interactive version]" not in html
    assert "<title>article</title>" in html


def test_main_converts_existing_root_articles(tmp_path: Path) -> None:
    (tmp_path / "ARTICLE_PART1_MEDIUM.md").write_text("# Part 1", encoding="utf-8")
    (tmp_path / "ARTICLE_PART2_MEDIUM.md").write_text("# Part 2", encoding="utf-8")

    md_to_html.main(tmp_path)

    assert (tmp_path / "ARTICLE_PART1_MEDIUM.html").exists()
    assert (tmp_path / "ARTICLE_PART2_MEDIUM.html").exists()


def test_resolve_image_base_url_uses_git_remote_and_commit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    responses = {
        ("config", "--get", "remote.origin.url"): "https://github.com/example/rayuela.git",
        ("rev-parse", "HEAD"): "abc123def",
    }

    def fake_run_git(args: tuple[str, ...] | list[str], cwd: Path) -> str | None:
        assert cwd == tmp_path
        return responses.get(tuple(args))

    monkeypatch.setattr(md_to_html, "run_git", fake_run_git)

    assert (
        md_to_html.resolve_image_base_url(tmp_path)
        == "https://raw.githubusercontent.com/example/rayuela/abc123def/article_images/"
    )


def test_resolve_image_base_url_prefers_explicit_override(tmp_path: Path) -> None:
    assert (
        md_to_html.resolve_image_base_url(tmp_path, "https://assets.example.test/images")
        == "https://assets.example.test/images/"
    )
