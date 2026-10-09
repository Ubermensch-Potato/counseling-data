"""
Fix audio chunk counts in post-train and post-valid English label JSONs.

Problem: In Korean standard files, some items within a section share the same
audio chunks (identical timestamps). When duplicates were removed during English
post-processing, items lost audio chunks. This script restores them.

Algorithm:
  For each section, group items by their Korean audio timestamp signature.
  Items sharing the same timestamps form a "group".
  In each group, English texts were split across items but should be shared.
  Collect all English texts from the group (in item order), map them to
  Korean positions (timestamp/type), and reconstruct all items identically.
"""

import json
import os
from pathlib import Path


BASE = Path("C:/Users/cwc/Desktop/cwc/TL/foundation data/adolescent_consultant_dataset")
POST_DIRS = {
    "post-train": BASE / "english_adolescent_consultant_dataset_corrected/post-train/labels",
    "post-valid": BASE / "english_adolescent_consultant_dataset_corrected/post-valid/labels",
}
KOR_DIRS = {
    "post-train": BASE / "korean_adolescent_consultant_dataset/train/labels",
    "post-valid": BASE / "korean_adolescent_consultant_dataset/valid/labels",
}
CATEGORIES = ["관찰필요", "상담필요", "응급", "정상군", "학대의심"]


def get_start_sig(audio_list):
    """Return tuple of start times as hashable signature."""
    return tuple(a["start"] for a in audio_list)


def fix_file(eng_path: Path, kor_path: Path) -> bool:
    """Fix audio chunks in one English file. Returns True if modified."""
    with open(eng_path, "r", encoding="utf-8") as f:
        eng = json.load(f)
    with open(kor_path, "r", encoding="utf-8") as f:
        kor = json.load(f)

    modified = False

    for sec_idx, (eng_sec, kor_sec) in enumerate(zip(eng["list"], kor["list"])):
        eng_items = eng_sec["list"]
        kor_items = kor_sec["list"]

        if len(eng_items) != len(kor_items):
            print(f"  WARN: item count mismatch in section {sec_idx} of {eng_path.name}")
            continue

        # Group items by their Korean audio timestamp signature
        sig_to_indices: dict = {}
        for i, kor_item in enumerate(kor_items):
            sig = get_start_sig(kor_item.get("audio", []))
            sig_to_indices.setdefault(sig, []).append(i)

        for sig, indices in sig_to_indices.items():
            if len(sig) == 0:
                continue  # No audio at all — nothing to fix

            # Check whether any item in this group needs fixing
            kor_template = kor_items[indices[0]]["audio"]
            needs_fix = any(
                len(eng_items[i].get("audio", [])) != len(kor_template)
                for i in indices
            )
            if not needs_fix:
                continue

            # Collect all English texts from the group in item order
            all_texts = []
            for i in indices:
                for entry in eng_items[i].get("audio", []):
                    all_texts.append(entry["text"])

            if len(all_texts) != len(kor_template):
                print(
                    f"  WARN: text count mismatch in {eng_path.name} "
                    f"section={eng_sec['문항']} sig={sig}: "
                    f"expected {len(kor_template)}, collected {len(all_texts)}"
                )
                continue

            # Reconstruct audio list from Korean structure + collected English texts
            reconstructed = []
            for pos, kor_entry in enumerate(kor_template):
                new_entry = {
                    "type": kor_entry["type"],
                    "text": all_texts[pos],
                    "wave": kor_entry["wave"],
                    "start": kor_entry["start"],
                    "end": kor_entry["end"],
                }
                reconstructed.append(new_entry)

            # Assign identical reconstructed audio to every item in the group
            for i in indices:
                eng_items[i]["audio"] = reconstructed
                modified = True

    if modified:
        with open(eng_path, "w", encoding="utf-8") as f:
            json.dump(eng, f, ensure_ascii=False, indent=4)

    return modified


def main():
    total_fixed = 0
    total_files = 0

    for split, eng_base in POST_DIRS.items():
        kor_base = KOR_DIRS[split]
        for cat in CATEGORIES:
            eng_cat = eng_base / cat
            kor_cat = kor_base / cat
            if not eng_cat.exists():
                continue

            for eng_file in sorted(eng_cat.glob("*.json")):
                kor_file = kor_cat / eng_file.name
                if not kor_file.exists():
                    print(f"MISSING Korean file: {kor_file}")
                    continue

                total_files += 1
                if fix_file(eng_file, kor_file):
                    total_fixed += 1

    print(f"\nDone. Fixed {total_fixed} / {total_files} files.")


if __name__ == "__main__":
    main()
