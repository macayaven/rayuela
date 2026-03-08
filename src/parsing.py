#!/usr/bin/env python3
"""
Parse Rayuela ePub into structured JSON for computational analysis.

Reads each of the 155 chapter XHTML files from the ePub, strips editorial
apparatus (chapter numbers, hopscotch navigation markers), and produces
clean prose text suitable for embedding.

Usage:
    python src/parsing.py

Input:  data/Rayuela - Julio Cortazar.epub
Output: data/rayuela_raw.json
"""

import html
import json
import re
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EPUB_PATH = PROJECT_ROOT / "data" / "Rayuela - Julio Cortazar.epub"
OUTPUT_PATH = PROJECT_ROOT / "data" / "rayuela_raw.json"

# Section boundaries (inclusive ranges)
# Part 1: chapters 1-36, Part 2: 37-56, Part 3: 57-155
SECTION_MAP = {
    "Del lado de allá": range(1, 37),
    "Del lado de acá": range(37, 57),
    "De otros lados (Capítulos prescindibles)": range(57, 156),
}


def get_section(chapter_number: int) -> str:
    """Return the section name for a given chapter number."""
    for section_name, chapter_range in SECTION_MAP.items():
        if chapter_number in chapter_range:
            return section_name
    raise ValueError(f"Chapter {chapter_number} not in any section")


# ---------------------------------------------------------------------------
# XHTML → plain text
# ---------------------------------------------------------------------------


def strip_chapter_xhtml(xhtml: str) -> str:
    """
    Extract clean prose from a chapter's XHTML.

    Removes:
      - <h3> chapter number heading (editorial, not prose)
      - <p class="derecha..."> navigation marker (hopscotch next-chapter link)
      - <p class="centrado"> with "* * *" (chapter 56's "Fin" stars)
      - ePub editor notes (*Nota edición epub:...) in chapters 34 and 96

    Preserves:
      - Paragraph structure (converted to double newlines)
      - Line breaks within stanzas/poetry (<br /> → newline)
      - Italics, bold content (tags stripped, text kept)
      - Blockquote/div content (tags stripped, text kept)
    """
    text = xhtml

    # 1. Remove the chapter number heading
    text = re.sub(r"<h3[^>]*>.*?</h3>", "", text, flags=re.DOTALL)

    # 2. Remove navigation marker (last <p> with class containing "derecha")
    text = re.sub(
        r'<p[^>]*class="[^"]*derecha[^"]*"[^>]*>.*?</p>',
        "",
        text,
        flags=re.DOTALL,
    )

    # 3. Remove the "* * *" marker from chapter 56
    text = re.sub(
        r'<p[^>]*class="centrado"[^>]*>\s*\*\s*\*\s*\*\s*</p>',
        "",
        text,
        flags=re.DOTALL,
    )

    # 4. Remove ePub editor notes (chapters 34 and 96)
    text = re.sub(
        r"<p[^>]*>\s*\(\*Nota edici[oó]n epub:.*?\).*?</p>",
        "",
        text,
        flags=re.DOTALL,
    )
    # Also remove empty <p> tags that sometimes precede editor notes
    text = re.sub(r"<p[^>]*>\s*</p>", "", text, flags=re.DOTALL)

    # 5. Strip everything outside <body>
    body_match = re.search(r"<body[^>]*>(.*)</body>", text, flags=re.DOTALL)
    if body_match:
        text = body_match.group(1)

    # 6. Convert <br /> to newline (poetry, line breaks)
    text = re.sub(r"<br\s*/?>", "\n", text)

    # 7. Convert closing block elements to paragraph breaks
    text = re.sub(r"</(?:p|div|blockquote)>", "\n\n", text)

    # 8. Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    # 9. Decode HTML entities (&nbsp; → space, &amp; → &, etc.)
    text = html.unescape(text)

    # 10. Normalize whitespace: strip each line, then collapse blank lines
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)  # max one blank line between paragraphs
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Tablero de Dirección (hopscotch reading sequence)
# ---------------------------------------------------------------------------


def parse_tablero(xhtml: str) -> list[int]:
    """
    Extract the hopscotch reading sequence from the Tablero XHTML.

    Returns a list of chapter numbers in hopscotch order.
    The sequence ends with 131-58-131 (the infinite loop).
    """
    # Get text content
    body = re.search(r"<body[^>]*>(.*)</body>", xhtml, flags=re.DOTALL)
    text = body.group(1) if body else xhtml

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    # Find the sequence lines: they contain numbers separated by dashes
    # Skip the prose paragraphs (Tablero explanation text)
    numbers: list[int] = []
    for line in text.split("\n"):
        # A sequence line has mostly numbers and dashes
        cleaned = line.strip()
        if not cleaned:
            continue
        # Check if this line is predominantly numbers and dashes
        tokens = re.findall(r"\d+", cleaned)
        if len(tokens) >= 3:  # sequence lines have many numbers
            numbers.extend(int(t) for t in tokens)

    return numbers


# ---------------------------------------------------------------------------
# Epigraph extraction
# ---------------------------------------------------------------------------


def extract_epigraph(xhtml: str) -> str:
    """Extract text from a section epigraph/header XHTML file."""
    body = re.search(r"<body[^>]*>(.*)</body>", xhtml, flags=re.DOTALL)
    if not body:
        return ""
    text = body.group(1)
    # Remove section headings (h2)
    text = re.sub(r"<h2[^>]*>.*?</h2>", "", text, flags=re.DOTALL)
    # Strip tags
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_epub(epub_path: Path) -> dict:
    """
    Parse the Rayuela ePub into a structured dictionary.

    Returns a dict matching the schema in RESEARCH_PLAN.md:
      metadata, reading_paths, sections, chapters
    """
    with zipfile.ZipFile(epub_path, "r") as z:
        # --- Tablero de Dirección ---
        tablero_xhtml = z.read("OEBPS/Text/TableroDeDireccion.xhtml").decode("utf-8")
        hopscotch_sequence = parse_tablero(tablero_xhtml)

        # --- Section epigraphs ---
        epigraphs: dict[str, Any] = {}

        # "Del lado de allá" — Vaché quote (0.xhtml)
        ep_alla = z.read("OEBPS/Text/0.xhtml").decode("utf-8")
        epigraphs["Del lado de allá"] = extract_epigraph(ep_alla)

        # "Del lado de acá" — Apollinaire quote
        ep_aca = z.read("OEBPS/Text/DelLadoDeAca.xhtml").decode("utf-8")
        epigraphs["Del lado de acá"] = extract_epigraph(ep_aca)

        # Front-matter epigraphs (before Part 1)
        ep_biblia = z.read("OEBPS/Text/000.xhtml").decode("utf-8")
        ep_bruto = z.read("OEBPS/Text/00.xhtml").decode("utf-8")
        epigraphs["front_matter"] = [
            extract_epigraph(ep_biblia),
            extract_epigraph(ep_bruto),
        ]

        # --- Chapters ---
        chapters = []
        for ch_num in range(1, 156):
            xhtml = z.read(f"OEBPS/Text/{ch_num}.xhtml").decode("utf-8")
            text = strip_chapter_xhtml(xhtml)

            chapter_data = {
                "number": ch_num,
                "section": get_section(ch_num),
                "text": text,
                "token_count": len(text.split()),  # rough word count; refine later
                "is_expendable": ch_num >= 57,
            }
            chapters.append(chapter_data)

    # --- Assemble output ---
    result = {
        "metadata": {
            "title": "Rayuela",
            "author": "Julio Cortázar",
            "year": 1963,
            "language": "es",
            "total_chapters": 155,
            "source": "ePub (Abraxas 14.12.11)",
            "parsing_notes": (
                "Editorial apparatus removed: chapter number headings and "
                "hopscotch navigation markers at chapter ends. Chapter 56's "
                "'* * *' (Fin marker) also removed. Chapter 55 has no "
                "navigation marker — it is the only chapter excluded from "
                "the hopscotch reading sequence."
            ),
        },
        "reading_paths": {
            "linear": list(range(1, 57)),
            "hopscotch": hopscotch_sequence,
        },
        "sections": {
            "Del lado de allá": {
                "chapters": list(range(1, 37)),
                "epigraph": epigraphs.get("Del lado de allá", ""),
            },
            "Del lado de acá": {
                "chapters": list(range(37, 57)),
                "epigraph": epigraphs.get("Del lado de acá", ""),
            },
            "De otros lados (Capítulos prescindibles)": {
                "chapters": list(range(57, 156)),
                "epigraph": None,
            },
        },
        "front_matter_epigraphs": epigraphs.get("front_matter", []),
        "chapters": chapters,
    }

    return result


def main(epub_path: Path = EPUB_PATH, output_path: Path = OUTPUT_PATH) -> int:
    """Parse the canonical Rayuela ePub and write the normalized JSON output."""
    if not epub_path.exists():
        print(f"Error: ePub not found at {epub_path}")
        return 1

    print(f"Parsing ePub: {epub_path.name}")
    data = parse_epub(epub_path)

    # Summary stats
    total_words = sum(ch["token_count"] for ch in data["chapters"])
    print(f"  Chapters parsed: {len(data['chapters'])}")
    print(f"  Total words (approx): {total_words:,}")
    print(f"  Hopscotch sequence length: {len(data['reading_paths']['hopscotch'])}")

    # Section breakdown
    for section_name, section_info in data["sections"].items():
        ch_nums = section_info["chapters"]
        section_words = sum(ch["token_count"] for ch in data["chapters"] if ch["number"] in ch_nums)
        print(f"  {section_name}: {len(ch_nums)} chapters, ~{section_words:,} words")

    # Shortest and longest chapters
    by_length = sorted(data["chapters"], key=lambda c: c["token_count"])
    print(f"  Shortest chapter: {by_length[0]['number']} ({by_length[0]['token_count']} words)")
    print(f"  Longest chapter:  {by_length[-1]['number']} ({by_length[-1]['token_count']} words)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nOutput written to: {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
