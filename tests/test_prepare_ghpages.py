from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import load_script_module

prepare_ghpages = load_script_module("prepare_ghpages_test", "scripts/prepare_ghpages.py")


def test_strip_inline_plotly_replaces_primary_bundle() -> None:
    html = '<script type="text/javascript">/**\n* plotly.js v3.3.1\npayload</script>'

    stripped = prepare_ghpages.strip_inline_plotly(html)

    assert prepare_ghpages.PLOTLY_CDN in stripped
    assert "plotly.js v3.3.1" not in stripped


def test_strip_inline_plotly_replaces_fallback_bundle() -> None:
    html = (
        '<script type="text/javascript">window.PlotlyConfig = {};</script>'
        '<script type="text/javascript">/**\n* plotly.js v3.3.1\npayload</script>'
    )

    stripped = prepare_ghpages.strip_inline_plotly(html)

    assert prepare_ghpages.PLOTLY_CDN in stripped


def test_strip_inline_plotly_leaves_unmatched_html_unchanged() -> None:
    html = "<html><body>No Plotly bundle</body></html>"
    assert prepare_ghpages.strip_inline_plotly(html) == html


def test_create_index_page_groups_expected_links() -> None:
    page = prepare_ghpages.create_index_page(
        ["article_heatmap.html", "umap_scale_a.html", "3d_scale_a.html"]
    )

    assert "Article Figures" in page
    assert "UMAP Projections" in page
    assert "3D Explorations" in page
    assert 'href="article_heatmap.html"' in page


def test_main_copies_selected_files_and_builds_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_dir = tmp_path / "figures"
    docs_dir = tmp_path / "docs"
    source_dir.mkdir()
    source_html = (
        '<script type="text/javascript">/**\n* plotly.js v3.3.1\npayload</script><div>figure</div>'
    )
    (source_dir / "article_heatmap.html").write_text(source_html, encoding="utf-8")

    monkeypatch.setattr(prepare_ghpages, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_ghpages, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(prepare_ghpages, "INCLUDE_FILES", ["article_heatmap.html"])

    prepare_ghpages.main()

    copied = (docs_dir / "article_heatmap.html").read_text(encoding="utf-8")
    index = (docs_dir / "index.html").read_text(encoding="utf-8")

    assert prepare_ghpages.PLOTLY_CDN in copied
    assert "plotly.js v3.3.1" not in copied
    assert 'href="article_heatmap.html"' in index


def test_main_skips_missing_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "figures"
    docs_dir = tmp_path / "docs"
    source_dir.mkdir()

    monkeypatch.setattr(prepare_ghpages, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_ghpages, "DOCS_DIR", docs_dir)
    monkeypatch.setattr(prepare_ghpages, "INCLUDE_FILES", ["missing.html"])

    prepare_ghpages.main()

    assert (docs_dir / "index.html").exists()


def test_extract_published_targets_parses_custom_domain_links() -> None:
    markdown = (
        "[Figure 1](https://carloscrespomacaya.com/rayuela/3d_scale_a.html)\n"
        "[Figure 2](https://carloscrespomacaya.com/rayuela/article_heatmap.html)\n"
        "[Repeat](https://carloscrespomacaya.com/rayuela/3d_scale_a.html)\n"
    )

    targets = prepare_ghpages.extract_published_targets(markdown)

    assert targets == ["3d_scale_a.html", "article_heatmap.html"]


def test_validate_published_targets_reports_missing_docs_and_bundle_entries(
    tmp_path: Path,
) -> None:
    article_path = tmp_path / "ARTICLE.md"
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    article_path.write_text(
        "[Figure](https://carloscrespomacaya.com/rayuela/article_heatmap.html)\n"
        "[Figure 2](https://carloscrespomacaya.com/rayuela/3d_scale_a.html)\n",
        encoding="utf-8",
    )
    (docs_dir / "article_heatmap.html").write_text("ok", encoding="utf-8")

    missing_docs, missing_publish_bundle = prepare_ghpages.validate_published_targets(
        article_paths=[article_path],
        docs_dir=docs_dir,
        include_files=["article_heatmap.html"],
    )

    assert missing_docs == ["3d_scale_a.html"]
    assert missing_publish_bundle == ["3d_scale_a.html"]


def test_repo_article_links_resolve_to_docs_bundle() -> None:
    missing_docs, missing_publish_bundle = prepare_ghpages.validate_published_targets()

    assert missing_docs == []
    assert missing_publish_bundle == []
