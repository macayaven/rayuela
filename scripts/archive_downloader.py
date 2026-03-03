#!/usr/bin/env python3
"""
Archive.org EPUB downloader for Latin American literary corpus.

Searches for Spanish-language editions of priority works and downloads EPUB files.

Usage:
    python scripts/archive_downloader.py

Output:
    Downloads EPUBs to data/ directory
"""

import requests
import time
import os
from pathlib import Path

# Known Archive.org identifiers for priority works
# These have been verified to have Spanish EPUBs
KNOWN_IDENTIFIERS = {
    # Priority works
    "garciamarquez_cienanos": "GabrielGarciaMarquezCienAnosDeSoledad",
    "borges_ficciones": "Ficciones0000borg",  # May need to try alternatives
    "borges_elaleph": None,  # Need to search
    "rulfo_pedroparamo": None,
    "bioycasares_invencionmorel": None,
    "fuentes_artemiocruz": None,
    "quiroga_cuentosamor": None,
    "cortazar_lospremios": None,
    "sabato_eltunel": None,
    "mutis_trampsteamer": None,
    "mutis_nievealmirante": None,
    "cabrerainfante_trestistestigres": None,
    "bolano_detectivessalvajes": None,
}

# Archive.org base URL for downloads
DOWNLOAD_BASE = "https://archive.org/download/{identifier}/{identifier}.epub"

# Search fallback URLs (for manual download if needed)
SEARCH_URLS = {
    "garciamarquez_cienanos": "https://archive.org/details/GabrielGarciaMarquezCienAnosDeSoledad",
    "borges_ficciones": "https://archive.org/details/ficciones0000borg",
    "borges_elaleph": "https://archive.org/search.php?query=creator%3A%22Jorge+Luis+Borges%22+AND+title%3A%22El+Aleph%22",
    "rulfo_pedroparamo": "https://archive.org/search.php?query=creator%3A%22Juan+Rulfo%22+AND+title%3A%22Pedro+Páramo%22",
    "bioycasares_invencionmorel": "https://archive.org/search.php?query=creator%3A%22Bioy+Casares%22+AND+title%3A%22La+invención+de+Morel%22",
    "fuentes_artemiocruz": "https://archive.org/search.php?query=creator%3A%22Carlos+Fuentes%22+AND+title%3A%22La+muerte+de+Artemio+Cruz%22",
    "quiroga_cuentosamor": "https://archive.org/search.php?query=creator%3A%22Horacio+Quiroga%22+AND+title%3A%22Cuentos%22",
    "cortazar_lospremios": "https://archive.org/search.php?query=creator%3A%22Julio+Cortázar%22+AND+title%3A%22Los+premios%22",
    "sabato_eltunel": "https://archive.org/search.php?query=creator%3A%22Ernesto+Sábato%22+AND+title%3A%22El+túnel%22",
    "mutis_trampsteamer": "https://archive.org/search.php?query=creator%3A%22Álvaro+Mutis%22+AND+title%3A%22Tramp+Steamer%22",
    "mutis_nievealmirante": "https://archive.org/search.php?query=creator%3A%22Álvaro+Mutis%22+AND+title%3A%22La+nieve+del+almirante%22",
    "cabrerainfante_trestistestigres": "https://archive.org/search.php?query=creator%3A%22Cabrera+Infante%22+AND+title%3A%22Tres+tristes+tigres%22",
    "bolano_detectivessalvajes": "https://archive.org/search.php?query=creator%3A%22Roberto+Bolaño%22+AND+title%3A%22Los+detectives+salvajes%22",
}


def check_epub_available(identifier):
    """Check if EPUB exists at Archive.org."""
    url = DOWNLOAD_BASE.format(identifier=identifier)
    try:
        response = requests.head(url, timeout=30, allow_redirects=True)
        return response.status_code == 200
    except Exception:
        return False


def download_epub(identifier, output_dir, custom_name=None):
    """Download EPUB from Archive.org."""
    url = DOWNLOAD_BASE.format(identifier=identifier)
    
    if custom_name:
        output_path = output_dir / custom_name
    else:
        output_path = output_dir / f"{identifier}.epub"
    
    try:
        print(f"  Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded: {output_path.name} ({size_mb:.1f} MB)")
        return output_path
    except Exception as e:
        print(f"  ✗ Error: {e}")
        if output_path.exists():
            output_path.unlink()
        return None


def main():
    """Main download workflow."""
    output_dir = Path("/home/carlos/education/rayuela/data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Archive.org EPUB Downloader")
    print("Latin American Literary Corpus")
    print("=" * 70)
    print()
    
    # Works to download (priority order)
    works = [
        # (key, author, title, custom_filename)
        ("garciamarquez_cienanos", "García Márquez", "Cien años de soledad", "garciamarquez_cienanos.epub"),
        ("borges_ficciones", "Borges", "Ficciones", "borges_ficciones.epub"),
        ("borges_elaleph", "Borges", "El Aleph", "borges_elaleph.epub"),
        ("rulfo_pedroparamo", "Rulfo", "Pedro Páramo", "rulfo_pedroparamo.epub"),
        ("bioycasares_invencionmorel", "Bioy Casares", "La invención de Morel", "bioycasares_invencionmorel.epub"),
        ("fuentes_artemiocruz", "Fuentes", "La muerte de Artemio Cruz", "fuentes_artemiocruz.epub"),
        ("quiroga_cuentosamor", "Quiroga", "Cuentos de amor de locura y de muerte", "quiroga_cuentosamor.epub"),
        ("cortazar_lospremios", "Cortázar", "Los premios", "cortazar_lospremios.epub"),
        ("sabato_eltunel", "Sábato", "El túnel", "sabato_eltunel.epub"),
        ("mutis_trampsteamer", "Mutis", "La última escala del Tramp Steamer", "mutis_trampsteamer.epub"),
        ("mutis_nievealmirante", "Mutis", "La nieve del almirante", "mutis_nievealmirante.epub"),
        ("cabrerainfante_trestistestigres", "Cabrera Infante", "Tres tristes tigres", "cabrerainfante_trestistestigres.epub"),
        ("bolano_detectivessalvajes", "Bolaño", "Los detectives salvajes", "bolano_detectivessalvajes.epub"),
    ]
    
    downloaded = []
    needs_manual = []
    
    for key, author, title, filename in works:
        print(f"\n{'='*60}")
        print(f"{author} — {title}")
        print(f"{'='*60}")
        
        # Check if already have
        existing = list(output_dir.glob(f"*{key.split('_')[1]}*.epub"))
        if existing:
            print(f"  Already have: {existing[0].name}")
            downloaded.append((author, title, str(existing[0]), "already had"))
            continue
        
        identifier = KNOWN_IDENTIFIERS.get(key)
        
        if identifier:
            # Try known identifier
            print(f"  Trying identifier: {identifier}")
            if check_epub_available(identifier):
                path = download_epub(identifier, output_dir, filename)
                if path:
                    downloaded.append((author, title, str(path), identifier))
                    time.sleep(2)
                    continue
        
        # No known identifier or download failed
        search_url = SEARCH_URLS.get(key, f"https://archive.org/search.php?query={key}")
        print(f"  Needs manual download")
        print(f"  Search URL: {search_url}")
        needs_manual.append((author, title, search_url))
        
        time.sleep(1)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Downloaded: {len(downloaded)}")
    print(f"Needs manual: {len(needs_manual)}")
    
    if downloaded:
        print("\n✓ Downloaded:")
        for author, title, path, source in downloaded:
            print(f"    {author} - {title}: {path}")
    
    if needs_manual:
        print("\n✗ Needs manual download (copy URL to browser):")
        for author, title, url in needs_manual:
            print(f"\n    {author} — {title}")
            print(f"    {url}")
    
    print("\n" + "=" * 70)
    print("INSTRUCTIONS FOR MANUAL DOWNLOAD:")
    print("=" * 70)
    print("""
1. Copy each URL above and paste into your browser
2. On the Archive.org item page, look for "EPUB" download option
3. Save the file to: /home/carlos/education/rayuela/data/
4. Rename using this pattern: {author}_{title}.epub
   Examples:
   - borges_elaleph.epub
   - rulfo_pedroparamo.epub
   - bioycasares_invencionmorel.epub

Alternative: Use Calibre's "Fetch News" or content server to search
and download directly.
    """)


if __name__ == "__main__":
    main()
