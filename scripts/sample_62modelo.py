#!/usr/bin/env python3
import os
import zipfile
import glob
from bs4 import BeautifulSoup

data_dir = "/home/carlos/education/rayuela/data"
os.chdir(data_dir)

# Get the exact filename using glob
files = glob.glob("62*.epub")
f = files[0]
print(f"Using file: {repr(f)}")

with zipfile.ZipFile(f, 'r') as z:
    # Read first few pages
    for page_num in [0, 1, 2, 10, 50]:
        page_file = f"EPUB/page_{page_num}.html"
        try:
            content = z.read(page_file).decode('utf-8')
            soup = BeautifulSoup(content, 'lxml')
            text = soup.get_text(separator='\n', strip=True)
            print(f"\n=== {page_file} ({len(text)} chars) ===")
            print(text[:500] if len(text) > 500 else text)
        except Exception as e:
            print(f"ERROR reading {page_file}: {e}")
