#!/usr/bin/env python3
import os
import zipfile
import glob

data_dir = "/home/carlos/education/rayuela/data"
os.chdir(data_dir)

# Get the exact filename using glob
files = glob.glob("62*.epub")
if not files:
    print("No 62 Modelo file found!")
    print("Available files:", glob.glob("*.epub"))
else:
    f = files[0]
    print(f"Using file: {repr(f)}")
    print(f"=== {f} ===")
    try:
        with zipfile.ZipFile(f, 'r') as z:
            names = z.namelist()
            # Show all xhtml/html/txt files
            text_files = [n for n in names if n.endswith(('.xhtml', '.html', '.txt', '.xml'))]
            print(f"Text files found: {len(text_files)}")
            for n in text_files:
                print(f"    {n}")
            
            # Also check image count
            image_files = [n for n in names if n.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
            print(f"\nImage files found: {len(image_files)}")
    except Exception as e:
        print(f"  ERROR: {e}")
