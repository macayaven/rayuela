#!/usr/bin/env python3
import os
import glob
import ebooklib
from ebooklib import epub

data_dir = "/home/carlos/education/rayuela/data"
os.chdir(data_dir)

# Find all epub files
for f in glob.glob("*.epub"):
    print(f"Found: {repr(f)}")
    print(f"  Size: {os.path.getsize(f)} bytes")
    try:
        book = epub.read_epub(f)
        print(f"  Title: {book.get_metadata('DC', 'title')}")
        print(f"  Author: {book.get_metadata('DC', 'creator')}")
        print(f"  Language: {book.get_metadata('DC', 'language')}")
        docs = [item for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT]
        print(f"  Document items: {len(docs)}")
        for d in docs[:5]:
            print(f"    - {d.get_name()}")
        if len(docs) > 5:
            print(f"    ... and {len(docs) - 5} more")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
