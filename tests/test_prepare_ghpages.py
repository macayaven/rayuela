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
