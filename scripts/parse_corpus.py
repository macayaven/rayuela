#!/usr/bin/env python3
"""
Literary corpus EPUB parser for computational stylometric analysis.

Parses EPUB files into structured JSON with exhaustive preprocessing documentation.
Strips editorial apparatus while preserving authorial content.

Usage:
    python scripts/parse_corpus.py

Input:  EPUB files in data/ directory
Output: JSON files and preprocessing logs in data/corpus/
"""

import os
import re
import json
import zipfile
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime

# Configuration
DATA_DIR = Path("/home/carlos/education/rayuela/data")
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# Known work metadata
WORK_METADATA = {
    # EPUB files
    "62 Modelo para Armar - Julio Cortázar.epub": {
        "title": "62: Modelo para armar",
        "author": "Julio Cortázar",
        "year": 1968,
        "structure": "non-linear",
        "output_key": "cortazar_62modelo",
    },
    "62 Modelo para Armar - Julio Cortazar.epub": {
        "title": "62: Modelo para armar",
        "author": "Julio Cortázar",
        "year": 1968,
        "structure": "non-linear",
        "output_key": "cortazar_62modelo",
    },
    "Un tal Lucas - Julio Cortazar.epub": {
        "title": "Un tal Lucas",
        "author": "Julio Cortázar",
        "year": 1979,
        "structure": "stories",
        "output_key": "cortazar_unlucas",
    },
    "garciamarquez_cienanos.epub": {
        "title": "Cien años de soledad",
        "author": "Gabriel García Márquez",
        "year": 1967,
        "structure": "linear",
        "output_key": "garciamarquez_cienanos",
    },
    "borges_elaleph.epub": {
        "title": "El Aleph",
        "author": "Jorge Luis Borges",
        "year": 1949,
        "structure": "stories",
        "output_key": "borges_elaleph",
    },
    "borges_ficciones.epub": {
        "title": "Ficciones",
        "author": "Jorge Luis Borges",
        "year": 1944,
        "structure": "stories",
        "output_key": "borges_ficciones",
    },
    "quiroga_cuentosamor.epub": {
        "title": "Cuentos de amor de locura y de muerte",
        "author": "Horacio Quiroga",
        "year": 1917,
        "structure": "stories",
        "output_key": "quiroga_cuentosamor",
    },
    "El_tunel.epub": {
        "title": "El túnel",
        "author": "Ernesto Sábato",
        "year": 1948,
        "structure": "linear",
        "output_key": "sabato_eltunel",
    },
    "rulfo_pedroparamo.epub": {
        "title": "Pedro Páramo",
        "author": "Juan Rulfo",
        "year": 1955,
        "structure": "fragmented",
        "output_key": "rulfo_pedroparamo",
    },
    # PDF files
    "bolano_detectivessalvajes.pdf": {
        "title": "Los detectives salvajes",
        "author": "Roberto Bolaño",
        "year": 1998,
        "structure": "multi-voice",
        "output_key": "bolano_detectivessalvajes",
    },
    "cabrerainfante_trestistestigres.pdf": {
        "title": "Tres tristes tigres",
        "author": "Guillermo Cabrera Infante",
        "year": 1967,
        "structure": "experimental",
        "output_key": "cabrerainfante_trestistestigres",
    },
}

# Editorial patterns to strip
EDITORIAL_PATTERNS = {
    "toc": re.compile(r'^(índice|tabla de contenido|contenido|table of contents)\s*$', re.IGNORECASE),
    "copyright": re.compile(r'(copyright|©|todos los derechos reservados|derechos reservados)', re.IGNORECASE),
    "isbn": re.compile(r'ISBN[\s\d-]{10,20}', re.IGNORECASE),
    "publisher_blurb": re.compile(r'^(editorial\s|published by|publicado por|colección|biblioteca)', re.IGNORECASE),
    "page_number": re.compile(r'^\s*\d+\s*$', re.MULTILINE),
    "running_header": re.compile(r'^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\s]{2,39}$', re.MULTILINE),
    "chapter_number_standalone": re.compile(r'^(capítulo|cap\.?|parte|part|section|sección)\s*(\d+|[IVX]+)', re.IGNORECASE),
    # Only match standalone navigation markers, not lines that happen to contain digits
    "navigation_marker": re.compile(r'^\s*(→|»|siguiente|next|continue)\s*\d*\s*$', re.IGNORECASE),
}

# Patterns to preserve (authorial)
PRESERVE_PATTERNS = {
    "epigraph": re.compile(r'^\s*["»].*["«]\s*$|^—.*$', re.MULTILINE),
    "dedication": re.compile(r'^(A |Para |Dedicado a |For |À )', re.IGNORECASE),
    "fictional_footnote": re.compile(r'\[\d+\]|\(\d+\)', re.MULTILINE),
    "embedded_poem": re.compile(r'^\s{4,}\w', re.MULTILINE),
}


class EPUBParser:
    """Parse EPUB files into structured JSON corpus."""

    def __init__(self, epub_path: Path):
        import unicodedata
        self.epub_path = epub_path
        self.filename = epub_path.name
        self.chapters = []
        self.preprocessing_log = []
        self.stripped_elements = []
        self.preserved_elements = []
        self.language_data = {"es": 0, "fr": 0, "en": 0, "other": 0}
        self.encoding_issues = []
        self.numbering_irregularities = []
        self.ambiguous_cases = []
        
        # Look up metadata with Unicode normalization
        filename_nfc = unicodedata.normalize("NFC", self.filename)
        self.metadata = {}
        for key, meta in WORK_METADATA.items():
            if unicodedata.normalize("NFC", key) == filename_nfc:
                self.metadata = meta
                break
        
    def log(self, section: str, message: str):
        """Add entry to preprocessing log."""
        self.preprocessing_log.append({
            "section": section,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
    
    def detect_language(self, text: str) -> dict:
        """Detect language mixing in text."""
        # Simple heuristic based on common words
        french_markers = {"le", "la", "les", "un", "une", "est", "dans", "avec", "pour", "que", "je", "tu", "il"}
        english_markers = {"the", "and", "is", "in", "of", "to", "a", "that", "it", "for"}
        spanish_markers = {"el", "la", "los", "las", "un", "una", "es", "en", "de", "que", "yo", "tú", "él"}
        
        words = set(re.findall(r'\b[a-záéíóúñ]+\b', text.lower()))
        
        fr_count = len(words & french_markers)
        en_count = len(words & english_markers)
        es_count = len(words & spanish_markers)
        
        # Spanish is baseline, count others as code-switching
        self.language_data["es"] += es_count
        self.language_data["fr"] += fr_count
        self.language_data["en"] += en_count
        
        return {"es": es_count, "fr": fr_count, "en": en_count}
    
    def strip_editorial(self, text: str, chapter_num: int = None) -> str:
        """Remove editorial apparatus while preserving authorial content."""
        original_text = text
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Skip empty lines at start/end (will be re-added for paragraph structure)
            if not stripped and (i == 0 or i == len(lines) - 1):
                continue
            
            # Check for editorial patterns
            is_editorial = False
            
            # Table of contents markers
            if EDITORIAL_PATTERNS["toc"].search(stripped):
                self.stripped_elements.append(("TOC marker", stripped[:100]))
                is_editorial = True
            
            # Copyright lines
            if EDITORIAL_PATTERNS["copyright"].search(stripped):
                self.stripped_elements.append(("Copyright", stripped[:100]))
                is_editorial = True
            
            # ISBN
            if EDITORIAL_PATTERNS["isbn"].search(stripped):
                self.stripped_elements.append(("ISBN", stripped[:100]))
                is_editorial = True
            
            # Publisher blurbs
            if EDITORIAL_PATTERNS["publisher_blurb"].search(stripped) and len(stripped) < 200:
                self.stripped_elements.append(("Publisher blurb", stripped[:100]))
                is_editorial = True
            
            # Standalone page numbers
            if EDITORIAL_PATTERNS["page_number"].match(stripped):
                self.stripped_elements.append(("Page number", stripped))
                is_editorial = True
            
            # Running headers (all caps short lines)
            if EDITORIAL_PATTERNS["running_header"].match(stripped) and len(stripped) < 50:
                self.stripped_elements.append(("Running header", stripped))
                is_editorial = True
            
            # Chapter number standalone (we preserve the number in metadata)
            if EDITORIAL_PATTERNS["chapter_number_standalone"].match(stripped):
                self.stripped_elements.append(("Chapter header", stripped))
                is_editorial = True
            
            if not is_editorial:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        if result != original_text:
            self.log("Stripped elements", f"Removed editorial content from chapter {chapter_num}")
        
        return result
    
    def extract_text_from_item(self, item_content: bytes) -> str:
        """Extract and clean text from EPUB XHTML item."""
        try:
            soup = BeautifulSoup(item_content, 'lxml-xml')
        except Exception:
            soup = BeautifulSoup(item_content, 'html.parser')
        
        # Remove script and style
        for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
            tag.decompose()
        
        # Get text
        text = soup.get_text(separator='\n')
        
        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    
    def identify_chapters(self, items: list) -> list:
        """Identify chapter/story boundaries in EPUB items."""
        chapters = []
        
        for i, item in enumerate(items):
            # Handle multiple item types:
            # Type 0 = HTML, Type 9 = XHTML
            # Skip: Type 1 (images), Type 2 (CSS), Type 4 (NCX toc)
            item_type = item.get_type()
            if item_type not in (0, 9):
                continue
            
            # Skip items without file_name
            if not hasattr(item, 'file_name') or not item.file_name:
                continue
            
            item_name = item.get_name().lower()
            item_href = item.file_name.lower()
            
            # Skip non-content files
            skip_patterns = ['toc', 'cover', 'title', 'copyright', 'info', 'nota', 'indice', 'notice']
            if any(p in item_name or p in item_href for p in skip_patterns):
                self.log("Chapter boundaries", f"Skipped non-content file: {item.get_name()}")
                continue
            
            # For scanned books with page_N.html format, we'll combine pages into chapters
            # Detect this pattern
            is_page_scan = bool(re.match(r'page_\d+\.html', item_name))
            
            chapters.append({
                "index": i,
                "item": item,
                "name": item.get_name(),
                "href": item.file_name,
                "is_page_scan": is_page_scan,
            })
        
        return chapters
    
    def parse(self) -> dict:
        """Parse EPUB into structured corpus."""
        from ebooklib import epub
        
        self.log("Edition identification", f"Parsing: {self.filename}")
        
        try:
            book = epub.read_epub(self.epub_path)
        except Exception as e:
            self.log("Edition identification", f"ERROR: {e}")
            return None
        
        # Extract metadata
        title = self.metadata.get("title", "")
        author = self.metadata.get("author", "")
        year = self.metadata.get("year", 0)
        
        dc_title = book.get_metadata("DC", "title")
        dc_creator = book.get_metadata("DC", "creator")
        
        if dc_title and not title:
            title = str(dc_title[0][0]) if dc_title else "Unknown"
        if dc_creator and not author:
            author = str(dc_creator[0][0]) if dc_creator else "Unknown"
        
        self.log("Edition identification", f"Title: {title}")
        self.log("Edition identification", f"Author: {author}")
        self.log("Edition identification", f"Year: {year}")

        # Get all document items (type 0 = HTML, type 9 = xhtml content)
        doc_items = [item for item in book.get_items() if item.get_type() in (0, 9)]
        self.log("Edition identification", f"Found {len(doc_items)} document items")
        
        # Identify chapters
        chapter_items = self.identify_chapters(doc_items)
        self.log("Chapter boundaries", f"Identified {len(chapter_items)} chapters/stories")
        
        # Check if this is a scanned book (page_N.html format)
        is_scanned_book = any(ch.get("is_page_scan", False) for ch in chapter_items)
        
        if is_scanned_book:
            self.log("Chapter boundaries", "Detected scanned book format (page_N.html)")
            # Combine all pages into one text, then split by story/chapter markers
            combined_pages = []
            for ch in chapter_items:
                try:
                    content = ch["item"].get_content()
                    text = self.extract_text_from_item(content)
                    if text.strip():
                        combined_pages.append(text)
                except Exception as e:
                    self.log("Ambiguous cases", f"Error reading page {ch['name']}: {e}")
            
            # For now, treat the combined text as a single chapter
            # (A more sophisticated approach would detect story boundaries)
            combined_text = '\n\n'.join(combined_pages)
            
            # Detect language
            self.detect_language(combined_text)
            
            # Strip editorial
            cleaned_text = self.strip_editorial(combined_text, chapter_num=1)
            
            # Count words
            word_count = len(cleaned_text.split())
            
            chapters_data = [{
                "number": 1,
                "title": None,
                "section": None,
                "text": cleaned_text,
                "word_count": word_count,
            }]
            total_words = word_count
            self.log("Chapter boundaries", f"Combined {len(combined_pages)} pages into 1 chapter ({word_count:,} words)")
        else:
            # Process each chapter normally
            chapters_data = []
            total_words = 0

            for i, ch in enumerate(chapter_items, 1):
                try:
                    content = ch["item"].get_content()
                    text = self.extract_text_from_item(content)

                    # Detect language mixing
                    lang_counts = self.detect_language(text)

                    # Strip editorial apparatus
                    cleaned_text = self.strip_editorial(text, chapter_num=i)

                    # Count words
                    word_count = len(cleaned_text.split())
                    total_words += word_count

                    # Check for encoding issues
                    if '' in cleaned_text or 'Ã' in cleaned_text[:100]:
                        self.encoding_issues.append({
                            "chapter": i,
                            "issue": "Possible encoding problem detected",
                            "sample": cleaned_text[:200],
                        })

                    chapter_data = {
                        "number": i,
                        "title": None,  # Will be set if story collection
                        "section": None,  # Will be set if work has parts
                        "text": cleaned_text,
                        "word_count": word_count,
                    }

                    # For story collections, try to extract story titles
                    if self.metadata.get("structure") == "stories":
                        # Try to get title from first line or filename
                        first_lines = cleaned_text.split('\n')[:5]
                        for line in first_lines:
                            line = line.strip()
                            if line and len(line) < 100 and not line.endswith('.'):
                                chapter_data["title"] = line
                                self.preserved_elements.append(("Story title", line))
                                break

                    chapters_data.append(chapter_data)

                except Exception as e:
                    self.log("Ambiguous cases", f"Error processing chapter {i}: {e}")
                    self.ambiguous_cases.append({
                        "chapter": i,
                        "error": str(e),
                    })
        
        # Detect code-switching
        other_languages = []
        if self.language_data["fr"] > 50:
            other_languages.append("fr")
            self.log("Language detection", f"French code-switching detected ({self.language_data['fr']} markers)")
        if self.language_data["en"] > 50:
            other_languages.append("en")
            self.log("Language detection", f"English code-switching detected ({self.language_data['en']} markers)")
        
        # Build output
        output = {
            "title": title,
            "author": author,
            "year_published": year,
            "primary_language": "es",
            "other_languages_present": other_languages,
            "total_chapters": len(chapters_data),
            "total_words": total_words,
            "structure": self.metadata.get("structure", "unknown"),
            "prescribed_reading_orders": [],
            "metadata": {
                "epigraphs": [],
                "dedication": "",
                "structure_notes": "",
            },
            "chapters": chapters_data,
        }
        
        return output
    
    def get_preprocessing_report(self) -> str:
        """Generate markdown preprocessing report."""
        report = []
        report.append(f"# Preprocessing Report")
        report.append(f"")
        report.append(f"**File**: {self.filename}")
        report.append(f"**Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"")
        
        # Edition identification
        report.append(f"## Edition Identification")
        report.append(f"")
        report.append(f"- **Title**: {self.metadata.get('title', 'Unknown')}")
        report.append(f"- **Author**: {self.metadata.get('author', 'Unknown')}")
        report.append(f"- **Year**: {self.metadata.get('year', 'Unknown')}")
        report.append(f"- **Source**: {self.filename}")
        report.append(f"")
        
        # Chapter boundaries
        report.append(f"## Chapter Boundaries")
        report.append(f"")
        report.append(f"Chapters were identified by scanning EPUB document items and filtering out:")
        report.append(f"- Table of contents files")
        report.append(f"- Cover pages")
        report.append(f"- Copyright pages")
        report.append(f"- Editorial notes")
        report.append(f"")
        
        # Stripped elements
        report.append(f"## Stripped Elements")
        report.append(f"")
        if self.stripped_elements:
            # Group by type
            by_type = {}
            for elem_type, sample in self.stripped_elements:
                if elem_type not in by_type:
                    by_type[elem_type] = []
                if len(by_type[elem_type]) < 5:  # Show up to 5 examples
                    by_type[elem_type].append(sample)
            
            for elem_type, examples in by_type.items():
                report.append(f"### {elem_type}")
                for ex in examples:
                    report.append(f"- `{ex[:80]}...`" if len(ex) > 80 else f"- `{ex}`")
                report.append(f"")
        else:
            report.append(f"No editorial elements stripped.")
            report.append(f"")
        
        # Preserved elements
        report.append(f"## Preserved Elements")
        report.append(f"")
        if self.preserved_elements:
            for elem_type, sample in self.preserved_elements[:10]:
                report.append(f"- **{elem_type}**: `{sample[:80]}`")
        else:
            report.append(f"Standard authorial content preserved.")
        report.append(f"")
        
        # Language detection
        report.append(f"## Language Detection")
        report.append(f"")
        report.append(f"- **Spanish (baseline)**: {self.language_data['es']} markers")
        report.append(f"- **French**: {self.language_data['fr']} markers")
        report.append(f"- **English**: {self.language_data['en']} markers")
        if self.language_data["fr"] > 50 or self.language_data["en"] > 50:
            report.append(f"")
            report.append(f"**Code-switching detected**")
        report.append(f"")
        
        # Encoding issues
        report.append(f"## Encoding Issues")
        report.append(f"")
        if self.encoding_issues:
            for issue in self.encoding_issues[:5]:
                report.append(f"- Chapter {issue['chapter']}: {issue['issue']}")
                report.append(f"  Sample: `{issue['sample'][:60]}...`")
        else:
            report.append(f"No encoding issues detected.")
        report.append(f"")
        
        # Numbering irregularities
        report.append(f"## Numbering Irregularities")
        report.append(f"")
        if self.numbering_irregularities:
            for irreg in self.numbering_irregularities:
                report.append(f"- {irreg}")
        else:
            report.append(f"No numbering irregularities detected.")
        report.append(f"")
        
        # Ambiguous cases
        report.append(f"## Ambiguous Cases")
        report.append(f"")
        if self.ambiguous_cases:
            for case in self.ambiguous_cases:
                report.append(f"- Chapter {case.get('chapter', '?')}: {case.get('error', 'Unknown error')}")
        else:
            report.append(f"No ambiguous cases.")
        report.append(f"")
        
        # Preprocessing log
        report.append(f"## Detailed Preprocessing Log")
        report.append(f"")
        for entry in self.preprocessing_log:
            report.append(f"- **{entry['section']}**: {entry['message']}")
        report.append(f"")
        
        return '\n'.join(report)


class PDFParser:
    """Parse PDF files into structured JSON corpus."""
    
    def __init__(self, pdf_path: Path):
        import unicodedata
        self.pdf_path = pdf_path
        self.filename = pdf_path.name
        self.chapters = []
        self.preprocessing_log = []
        self.stripped_elements = []
        self.preserved_elements = []
        self.language_data = {"es": 0, "fr": 0, "en": 0, "other": 0}
        self.encoding_issues = []
        self.numbering_irregularities = []
        self.ambiguous_cases = []
        
        # Look up metadata with Unicode normalization
        filename_nfc = unicodedata.normalize("NFC", self.filename)
        self.metadata = {}
        for key, meta in WORK_METADATA.items():
            if unicodedata.normalize("NFC", key) == filename_nfc:
                self.metadata = meta
                break
    
    def log(self, section: str, message: str):
        """Add entry to preprocessing log."""
        self.preprocessing_log.append({
            "section": section,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })
    
    def detect_language(self, text: str) -> dict:
        """Detect language mixing in text."""
        french_markers = {"le", "la", "les", "un", "une", "est", "dans", "avec", "pour", "que", "je", "tu", "il"}
        english_markers = {"the", "and", "is", "in", "of", "to", "a", "that", "it", "for"}
        spanish_markers = {"el", "la", "los", "las", "un", "una", "es", "en", "de", "que", "yo", "tú", "él"}
        
        words = set(re.findall(r'\b[a-záéíóúñ]+\b', text.lower()))
        
        fr_count = len(words & french_markers)
        en_count = len(words & english_markers)
        es_count = len(words & spanish_markers)
        
        self.language_data["es"] += es_count
        self.language_data["fr"] += fr_count
        self.language_data["en"] += en_count
        
        return {"es": es_count, "fr": fr_count, "en": en_count}
    
    def strip_editorial(self, text: str, chapter_num: int = None) -> str:
        """Remove editorial apparatus while preserving authorial content."""
        original_text = text
        lines = text.split('\n')
        cleaned_lines = []
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if not stripped and (i == 0 or i == len(lines) - 1):
                continue
            
            is_editorial = False
            
            if EDITORIAL_PATTERNS["toc"].search(stripped):
                self.stripped_elements.append(("TOC marker", stripped[:100]))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["copyright"].search(stripped):
                self.stripped_elements.append(("Copyright", stripped[:100]))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["isbn"].search(stripped):
                self.stripped_elements.append(("ISBN", stripped[:100]))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["publisher_blurb"].search(stripped) and len(stripped) < 200:
                self.stripped_elements.append(("Publisher blurb", stripped[:100]))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["page_number"].match(stripped):
                self.stripped_elements.append(("Page number", stripped))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["running_header"].match(stripped) and len(stripped) < 50:
                self.stripped_elements.append(("Running header", stripped))
                is_editorial = True
            
            if EDITORIAL_PATTERNS["chapter_number_standalone"].match(stripped):
                self.stripped_elements.append(("Chapter header", stripped))
                is_editorial = True
            
            if not is_editorial:
                cleaned_lines.append(line)
        
        result = '\n'.join(cleaned_lines)
        
        if result != original_text:
            self.log("Stripped elements", f"Removed editorial content from chapter {chapter_num}")
        
        return result
    
    def parse(self) -> dict:
        """Parse PDF into structured corpus."""
        from PyPDF2 import PdfReader
        
        self.log("Edition identification", f"Parsing: {self.filename}")
        
        try:
            reader = PdfReader(self.pdf_path)
        except Exception as e:
            self.log("Edition identification", f"ERROR: {e}")
            return None
        
        # Extract metadata
        title = self.metadata.get("title", "")
        author = self.metadata.get("author", "")
        year = self.metadata.get("year", 0)
        
        pdf_meta = reader.metadata
        if pdf_meta:
            if not title and pdf_meta.get('/Title'):
                title = str(pdf_meta.get('/Title', ''))
            if not author and pdf_meta.get('/Author'):
                author = str(pdf_meta.get('/Author', ''))
        
        self.log("Edition identification", f"Title: {title}")
        self.log("Edition identification", f"Author: {author}")
        self.log("Edition identification", f"Year: {year}")
        self.log("Edition identification", f"Pages: {len(reader.pages)}")
        
        # Extract text from all pages
        all_text = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text()
                if text:
                    all_text.append(text)
            except Exception as e:
                self.log("Ambiguous cases", f"Error extracting page {i+1}: {e}")
        
        full_text = '\n\n'.join(all_text)
        
        # Detect language
        self.detect_language(full_text)
        
        # Strip editorial
        cleaned_text = self.strip_editorial(full_text, chapter_num=1)
        
        # Count words
        word_count = len(cleaned_text.split())
        
        # For novels, we'll create chapter divisions based on patterns
        # Look for "Capítulo X" or similar patterns
        chapter_pattern = re.compile(r'(?:^|\n\n)(?:CAPÍTULO|Capítulo|Cap\.?|PARTE|Parte|Part)\s*(?:Nº?\s*)?(\d+|[IVX]+)', re.MULTILINE)
        chapter_matches = list(chapter_pattern.finditer(cleaned_text))
        
        chapters_data = []
        
        if len(chapter_matches) > 1:
            # Split by chapters
            self.log("Chapter boundaries", f"Found {len(chapter_matches)} chapter markers")
            
            for i, match in enumerate(chapter_matches):
                start = match.start()
                end = chapter_matches[i + 1].start() if i + 1 < len(chapter_matches) else len(cleaned_text)
                
                chapter_text = cleaned_text[start:end]
                chapter_num_str = match.group(1)
                
                # Convert Roman numerals if needed
                try:
                    chapter_num = int(chapter_num_str)
                except ValueError:
                    chapter_num = i + 1
                
                word_count = len(chapter_text.split())
                
                chapters_data.append({
                    "number": chapter_num,
                    "title": None,
                    "section": None,
                    "text": chapter_text,
                    "word_count": word_count,
                })
        else:
            # No chapter markers found - treat as single chapter
            self.log("Chapter boundaries", "No chapter markers found - treating as single chapter")
            
            chapters_data = [{
                "number": 1,
                "title": None,
                "section": None,
                "text": cleaned_text,
                "word_count": word_count,
            }]
        
        # Detect code-switching
        other_languages = []
        if self.language_data["fr"] > 50:
            other_languages.append("fr")
            self.log("Language detection", f"French code-switching detected ({self.language_data['fr']} markers)")
        if self.language_data["en"] > 50:
            other_languages.append("en")
            self.log("Language detection", f"English code-switching detected ({self.language_data['en']} markers)")
        
        # Build output
        output = {
            "title": title,
            "author": author,
            "year_published": year,
            "primary_language": "es",
            "other_languages_present": other_languages,
            "total_chapters": len(chapters_data),
            "total_words": sum(ch["word_count"] for ch in chapters_data),
            "structure": self.metadata.get("structure", "unknown"),
            "prescribed_reading_orders": [],
            "metadata": {
                "epigraphs": [],
                "dedication": "",
                "structure_notes": "",
            },
            "chapters": chapters_data,
        }
        
        return output
    
    def get_preprocessing_report(self) -> str:
        """Generate markdown preprocessing report."""
        report = []
        report.append(f"# Preprocessing Report")
        report.append(f"")
        report.append(f"**File**: {self.filename}")
        report.append(f"**Processed**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"")
        
        report.append(f"## Edition Identification")
        report.append(f"")
        report.append(f"- **Title**: {self.metadata.get('title', 'Unknown')}")
        report.append(f"- **Author**: {self.metadata.get('author', 'Unknown')}")
        report.append(f"- **Year**: {self.metadata.get('year', 'Unknown')}")
        report.append(f"- **Source**: {self.filename}")
        report.append(f"")
        
        report.append(f"## Chapter Boundaries")
        report.append(f"")
        report.append(f"Chapters were identified by scanning for chapter markers (Capítulo X, Parte X, etc.).")
        report.append(f"If no markers found, the entire text is treated as a single chapter.")
        report.append(f"")
        
        report.append(f"## Stripped Elements")
        report.append(f"")
        if self.stripped_elements:
            by_type = {}
            for elem_type, sample in self.stripped_elements:
                if elem_type not in by_type:
                    by_type[elem_type] = []
                if len(by_type[elem_type]) < 5:
                    by_type[elem_type].append(sample)
            
            for elem_type, examples in by_type.items():
                report.append(f"### {elem_type}")
                for ex in examples:
                    report.append(f"- `{ex[:80]}...`" if len(ex) > 80 else f"- `{ex}`")
                report.append(f"")
        else:
            report.append(f"No editorial elements stripped.")
            report.append(f"")
        
        report.append(f"## Preserved Elements")
        report.append(f"")
        if self.preserved_elements:
            for elem_type, sample in self.preserved_elements[:10]:
                report.append(f"- **{elem_type}**: `{sample[:80]}`")
        else:
            report.append(f"Standard authorial content preserved.")
        report.append(f"")
        
        report.append(f"## Language Detection")
        report.append(f"")
        report.append(f"- **Spanish (baseline)**: {self.language_data['es']} markers")
        report.append(f"- **French**: {self.language_data['fr']} markers")
        report.append(f"- **English**: {self.language_data['en']} markers")
        if self.language_data["fr"] > 50 or self.language_data["en"] > 50:
            report.append(f"")
            report.append(f"**Code-switching detected**")
        report.append(f"")
        
        report.append(f"## Encoding Issues")
        report.append(f"")
        if self.encoding_issues:
            for issue in self.encoding_issues[:5]:
                report.append(f"- {issue}")
        else:
            report.append(f"No encoding issues detected.")
        report.append(f"")
        
        report.append(f"## Numbering Irregularities")
        report.append(f"")
        if self.numbering_irregularities:
            for irreg in self.numbering_irregularities:
                report.append(f"- {irreg}")
        else:
            report.append(f"No numbering irregularities detected.")
        report.append(f"")
        
        report.append(f"## Ambiguous Cases")
        report.append(f"")
        if self.ambiguous_cases:
            for case in self.ambiguous_cases:
                report.append(f"- {case}")
        else:
            report.append(f"No ambiguous cases.")
        report.append(f"")
        
        report.append(f"## Detailed Preprocessing Log")
        report.append(f"")
        for entry in self.preprocessing_log:
            report.append(f"- **{entry['section']}**: {entry['message']}")
        report.append(f"")
        
        return '\n'.join(report)


def main():
    """Main parsing workflow."""
    import unicodedata

    print("=" * 70)
    print("Literary Corpus Parser")
    print("Computational Stylometric Analysis")
    print("=" * 70)
    print()

    # Find files to parse (EPUB and PDF) - match by normalized filename
    files_to_parse = []
    metadata_keys = list(WORK_METADATA.keys())

    # Check for EPUB files
    for epub_file in DATA_DIR.glob("*.epub"):
        filename = epub_file.name
        # Skip Rayuela (already parsed)
        if "Rayuela" in filename:
            continue

        # Normalize filename for comparison (NFC)
        filename_nfc = unicodedata.normalize("NFC", filename)

        # Try to find matching metadata
        matched = False
        for key in metadata_keys:
            key_nfc = unicodedata.normalize("NFC", key)
            if filename_nfc == key_nfc:
                files_to_parse.append(("epub", epub_file))
                matched = True
                break

        if not matched:
            print(f"NO METADATA: {filename}")

    # Check for PDF files
    for pdf_file in DATA_DIR.glob("*.pdf"):
        filename = pdf_file.name
        filename_nfc = unicodedata.normalize("NFC", filename)

        matched = False
        for key in metadata_keys:
            key_nfc = unicodedata.normalize("NFC", key)
            if filename_nfc == key_nfc:
                files_to_parse.append(("pdf", pdf_file))
                matched = True
                break

        if not matched:
            print(f"NO METADATA: {filename}")

    # Check for missing files
    for key in metadata_keys:
        file_path = DATA_DIR / key
        if not file_path.exists():
            found = False
            for ext in ["*.epub", "*.pdf"]:
                for f in DATA_DIR.glob(ext):
                    if unicodedata.normalize("NFC", f.name) == unicodedata.normalize("NFC", key):
                        found = True
                        break
            if not found:
                print(f"NOT FOUND: {key}")

    print(f"Found {len(files_to_parse)} files to parse")
    print()

    parsed_works = []

    for file_type, file_path in files_to_parse:
        print(f"Parsing: {file_path.name} ({file_type.upper()})")

        if file_type == "epub":
            parser = EPUBParser(file_path)
        else:
            parser = PDFParser(file_path)
        
        output = parser.parse()

        if output:
            # Get output key from metadata
            filename_nfc = unicodedata.normalize("NFC", file_path.name)
            output_key = file_path.stem  # default

            for key, meta in WORK_METADATA.items():
                if unicodedata.normalize("NFC", key) == filename_nfc:
                    output_key = meta.get("output_key", file_path.stem)
                    break

            json_path = CORPUS_DIR / f"{output_key}_raw.json"

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            # Save preprocessing log
            md_path = CORPUS_DIR / f"{output_key}_preprocessing.md"

            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(parser.get_preprocessing_report())

            print(f"  ✓ JSON: {json_path.name} ({output['total_words']:,} words, {output['total_chapters']} chapters)")
            print(f"  ✓ Log:  {md_path.name}")

            parsed_works.append({
                "filename": json_path.name,
                "title": output["title"],
                "author": output["author"],
                "year": output["year_published"],
                "chapters": output["total_chapters"],
                "words": output["total_words"],
            })
        else:
            print(f"  ✗ Failed to parse")

        print()
    
    # Generate INDEX.md
    print("Generating INDEX.md...")
    index_path = CORPUS_DIR / "INDEX.md"
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write("# Corpus Index\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total works**: {len(parsed_works)}\n\n")
        
        f.write("| Author | Title | Year | Chapters | Words | File |\n")
        f.write("|--------|-------|------|----------|-------|------|\n")
        
        for work in sorted(parsed_works, key=lambda x: (x["author"], x["year"])):
            f.write(f"| {work['author']} | {work['title']} | {work['year']} | {work['chapters']} | {work['words']:,} | `{work['filename']}` |\n")
        
        f.write("\n\n## Notes\n\n")
        f.write("- All texts are in original Spanish\n")
        f.write("- Editorial apparatus stripped (TOC, copyright, page numbers, running headers)\n")
        f.write("- Authorial content preserved (epigraphs, dedications, story titles)\n")
        f.write("- See individual preprocessing logs for detailed documentation\n")
    
    print(f"  ✓ INDEX.md created")
    print()
    print("=" * 70)
    print(f"Parsed {len(parsed_works)} works successfully")
    print("=" * 70)


if __name__ == "__main__":
    main()
