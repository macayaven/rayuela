from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import parsing


def test_get_section_maps_known_ranges() -> None:
    assert parsing.get_section(1) == "Del lado de allá"
    assert parsing.get_section(37) == "Del lado de acá"
    assert parsing.get_section(155) == "De otros lados (Capítulos prescindibles)"


def test_get_section_rejects_invalid_chapter() -> None:
    with pytest.raises(ValueError):
        parsing.get_section(0)


def test_strip_chapter_xhtml_removes_editorial_artifacts() -> None:
    xhtml = """
    <html><body>
      <h3>12</h3>
      <p>Primer párrafo<br />con salto.</p>
      <div><em>Segundo</em> párrafo.</div>
      <p class="derecha">73</p>
      <p class="centrado">* * *</p>
      <p>(*Nota edición epub: quitar esto)</p>
    </body></html>
    """

    cleaned = parsing.strip_chapter_xhtml(xhtml)

    assert "73" not in cleaned
    assert "Nota edición epub" not in cleaned
    assert "Primer párrafo\ncon salto." in cleaned
    assert "Segundo párrafo." in cleaned


def test_strip_chapter_xhtml_without_body_still_returns_text() -> None:
    cleaned = parsing.strip_chapter_xhtml("<p>Texto suelto</p>")
    assert cleaned == "Texto suelto"


def test_parse_tablero_extracts_numeric_sequence() -> None:
    xhtml = """
    <html><body>
      <p>Tablero explanatory prose.</p>
      73-1-2-116-3
      <div>84-4-71</div>
    </body></html>
    """

    assert parsing.parse_tablero(xhtml) == [73, 1, 2, 116, 3, 84, 4, 71]


def test_extract_epigraph_removes_heading_and_tags() -> None:
    xhtml = """
    <html><body>
      <h2>Del lado de allá</h2>
      <p>Linea uno<br/>Linea dos</p>
    </body></html>
    """

    assert parsing.extract_epigraph(xhtml) == "Linea uno\nLinea dos"


def test_extract_epigraph_without_body_returns_empty_string() -> None:
    assert parsing.extract_epigraph("<div>Sin body</div>") == ""


def test_parse_epub_builds_expected_shape(tmp_path: Path) -> None:
    epub_path = tmp_path / "mini.epub"
    chapter_xhtml = (
        "<html><body><h3>1</h3><p>Texto base.</p>"
        "<p class='derecha'>73</p></body></html>"
    )
    tablero = "<html><body><p>Intro</p>\n73-1-2-116-3\n84-4-71</body></html>"
    epigraph = "<html><body><h2>Titulo</h2><p>Epígrafe</p></body></html>"

    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("OEBPS/Text/TableroDeDireccion.xhtml", tablero)
        archive.writestr("OEBPS/Text/0.xhtml", epigraph)
        archive.writestr("OEBPS/Text/000.xhtml", epigraph)
        archive.writestr("OEBPS/Text/00.xhtml", epigraph)
        archive.writestr("OEBPS/Text/DelLadoDeAca.xhtml", epigraph)
        for chapter in range(1, 156):
            archive.writestr(f"OEBPS/Text/{chapter}.xhtml", chapter_xhtml)

    parsed = parsing.parse_epub(epub_path)

    assert parsed["metadata"]["total_chapters"] == 155
    assert parsed["reading_paths"]["linear"][:3] == [1, 2, 3]
    assert parsed["reading_paths"]["hopscotch"][:4] == [73, 1, 2, 116]
    assert len(parsed["chapters"]) == 155
    assert parsed["chapters"][0]["section"] == "Del lado de allá"
    assert parsed["chapters"][-1]["is_expendable"] is True
    assert parsed["sections"]["Del lado de acá"]["epigraph"] == "Epígrafe"

    json.dumps(parsed, ensure_ascii=False)


def test_main_writes_output_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    epub_path = tmp_path / "mini.epub"
    output_path = tmp_path / "parsed.json"
    chapter_xhtml = (
        "<html><body><h3>1</h3><p>Texto base.</p>"
        "<p class='derecha'>73</p></body></html>"
    )
    tablero = "<html><body><p>Intro</p>\n73-1-2-116-3\n84-4-71</body></html>"
    epigraph = "<html><body><h2>Titulo</h2><p>Epígrafe</p></body></html>"

    with zipfile.ZipFile(epub_path, "w") as archive:
        archive.writestr("OEBPS/Text/TableroDeDireccion.xhtml", tablero)
        archive.writestr("OEBPS/Text/0.xhtml", epigraph)
        archive.writestr("OEBPS/Text/000.xhtml", epigraph)
        archive.writestr("OEBPS/Text/00.xhtml", epigraph)
        archive.writestr("OEBPS/Text/DelLadoDeAca.xhtml", epigraph)
        for chapter in range(1, 156):
            archive.writestr(f"OEBPS/Text/{chapter}.xhtml", chapter_xhtml)

    exit_code = parsing.main(epub_path, output_path)
    captured = capsys.readouterr()

    assert exit_code == 0
    assert output_path.exists()
    assert "Chapters parsed: 155" in captured.out


def test_main_returns_nonzero_when_epub_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = parsing.main(tmp_path / "missing.epub", tmp_path / "out.json")
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error: ePub not found" in captured.out
