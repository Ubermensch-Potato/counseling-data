"""
Scan all .txt source files in train/inputs/ and valid/inputs/ for non-English
characters and write the results to error_log.json as [filename, [line_numbers]] tuples.

A character is flagged if it is:
  - Outside ASCII (0x00-0x7F), AND
  - A letter or number (Unicode category L* or N*) — i.e. belongs to a language script.
Punctuation and symbols (P*, S*) are allowed regardless of Unicode block,
since characters like curly quotes or em dashes are not language-specific.
"""

import os
import re
import json
import unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))

NON_ASCII = re.compile(r'[^\x00-\x7F]')


def is_language_char(ch):
    """Return True if ch is a non-ASCII letter or digit (i.e. belongs to a foreign script)."""
    cat = unicodedata.category(ch)
    return cat[0] in ('L', 'N')


def has_non_english(line):
    return any(is_language_char(ch) for ch in NON_ASCII.findall(line))


def scan():
    error_files = []

    for split in ["train", "valid"]:
        input_dir = os.path.join(BASE, split, "inputs")
        for root, dirs, files in os.walk(input_dir):
            for fname in sorted(files):
                if not fname.endswith(".txt"):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                    bad_lines = [i + 1 for i, l in enumerate(lines) if has_non_english(l)]
                    if bad_lines:
                        rel = os.path.relpath(fpath, BASE).replace("\\", "/")
                        error_files.append([rel, bad_lines])
                except Exception as e:
                    print(f"ERROR reading {fpath}: {e}")

    out_path = os.path.join(BASE, "non_eng_error_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"error_files": error_files}, f, ensure_ascii=False, indent=2)
    print(f"Written {len(error_files)} entries to non_eng_error_log.json")


if __name__ == "__main__":
    scan()
