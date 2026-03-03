#!/usr/bin/env python3
import os
import zipfile
import glob

data_dir = "/home/carlos/education/rayuela/data"
os.chdir(data_dir)

for f in glob.glob("*.epub"):
    print(f"=== {f} ===")
    try:
        with zipfile.ZipFile(f, 'r') as z:
            names = z.namelist()
            print(f"  Total files: {len(names)}")
            # Show first 20 files
            for n in names[:20]:
                print(f"    {n}")
            if len(names) > 20:
                print(f"    ... and {len(names) - 20} more")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
