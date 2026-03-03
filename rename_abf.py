"""
Rename ABF files from compact pClamp format to underscore-separated format.

Example: 26130000.abf  ->  2026_01_30_0000.abf
         Format: YYMDDNNN.abf -> YYYY_MM_DD_0NNN.abf

Where:
  YY  = 2-digit year (prefixed with 20)
  M   = 1-digit month (1-9) or letter (A/O=Oct, B/N=Nov, C/D=Dec)
  DD  = 2-digit day
  NNN = 3-digit run number
"""

import os
import re
import sys
from pathlib import Path


def parse_compact_name(filename: str):
    """
    Parse compact pClamp ABF filename like 26130000.abf
    Format: YYMDDNNN (2+1+2+3 = 8 characters)
    Returns (new_name, year, month, day, run_num) or None if not matching.
    """
    base = Path(filename).stem
    ext = Path(filename).suffix

    # Match: 2-digit year, 1-char month (digit or letter), 2-digit day, 3-digit run
    m = re.match(r'^(\d{2})([1-9A-Ca-cOoNnDd])(\d{2})(\d{3})$', base)
    if not m:
        return None

    yy, month_char, dd, run_num = m.groups()
    year = f"20{yy}"

    # Convert month character to 2-digit month string
    month_map = {
        'A': '10', 'a': '10', 'O': '10', 'o': '10',
        'B': '11', 'b': '11', 'N': '11', 'n': '11',
        'C': '12', 'c': '12', 'D': '12', 'd': '12',
    }
    if month_char.isdigit():
        mm = f"0{month_char}"  # 1-9 -> 01-09
    else:
        mm = month_map.get(month_char)
        if mm is None:
            return None

    # Validate day
    day = int(dd)
    if day < 1 or day > 31:
        return None

    new_name = f"{year}_{mm}_{dd}_{run_num}{ext}"
    return new_name, year, mm, dd, run_num


def main():
    # Prompt for directory
    while True:
        directory = input("Enter the directory path containing ABF files to rename: ").strip()
        if not directory:
            print("  Please enter a path.")
            continue
        if not os.path.isdir(directory):
            print(f"  Directory not found: {directory}")
            continue
        break

    abf_files = sorted(Path(directory).glob("*.abf"))
    if not abf_files:
        print(f"No .abf files found in {directory}")
        return

    # Preview renames
    renames = []
    skipped = []
    for f in abf_files:
        result = parse_compact_name(f.name)
        if result:
            new_name = result[0]
            new_path = f.parent / new_name
            renames.append((f, new_path))
        else:
            skipped.append(f.name)

    if not renames:
        print("No files matched the compact format (YYMMDDnnnn.abf).")
        if skipped:
            print(f"Skipped {len(skipped)} file(s) with unrecognized format:")
            for s in skipped:
                print(f"  {s}")
        return

    # Show preview
    print(f"\nFound {len(renames)} file(s) to rename:\n")
    for old, new in renames:
        print(f"  {old.name}  ->  {new.name}")

    if skipped:
        print(f"\nSkipping {len(skipped)} file(s) with unrecognized format:")
        for s in skipped:
            print(f"  {s}")

    # Confirm
    confirm = input(f"\nProceed with renaming {len(renames)} file(s)? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return

    # Rename
    success = 0
    for old, new in renames:
        if new.exists():
            print(f"  [SKIP] {new.name} already exists, skipping {old.name}")
            continue
        try:
            old.rename(new)
            print(f"  [OK] {old.name} -> {new.name}")
            success += 1
        except Exception as e:
            print(f"  [ERROR] {old.name}: {e}")

    print(f"\nDone. Renamed {success}/{len(renames)} file(s).")


if __name__ == "__main__":
    main()
