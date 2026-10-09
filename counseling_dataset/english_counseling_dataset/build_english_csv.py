"""
Build english_train.csv and english_valid.csv from the translated English counseling dataset.

Directory layout mirrors korean_counseling_dataset/:
  train/inputs/  -> flat .txt files  OR  TS_*/ subdirs containing .txt files
  train/labels/  -> flat .json files OR  TL_*/ subdirs containing .json files
  valid/inputs/  -> flat .txt files  OR  VS_*/ subdirs containing .txt files
  valid/labels/  -> flat .json files OR  VL_*/ subdirs containing .json files

Input->Label filename mapping:
  resource_X.txt  <->  label_X.json
"""

import os
import re
import json
import pandas as pd


def read_json_labels(json_path):
    """Extract all top-level label fields. Falls back to regex for malformed JSON."""
    TOP_LEVEL_FIELDS = ["id", "age", "gender", "depression", "anxiety", "addiction", "class", "summary"]
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return {k: d.get(k) for k in TOP_LEVEL_FIELDS}
    except Exception:
        with open(json_path, "r", encoding="utf-8") as f:
            head = "".join(f.readline() for _ in range(30))
        result = {}
        for field in ["depression", "anxiety", "addiction", "age"]:
            m = re.search(rf'"{field}"\s*:\s*(\d+)', head)
            result[field] = int(m.group(1)) if m else None
        for field in ["id", "gender", "class"]:
            m = re.search(rf'"{field}"\s*:\s*"([^"]*)"', head)
            result[field] = m.group(1) if m else None
        m = re.search(r'"summary"\s*:\s*"(.*?)"(?:\s*,|\s*\n)', head, re.DOTALL)
        result["summary"] = m.group(1) if m else None
        return result


def collect_sessions(split_dir):
    """Walk split_dir/inputs and yield (txt_path, json_path) pairs."""
    input_dir = os.path.join(split_dir, "inputs")
    label_dir = os.path.join(split_dir, "labels")
    pairs = []

    for item in sorted(os.listdir(input_dir)):
        item_path = os.path.join(input_dir, item)

        if os.path.isfile(item_path) and item.endswith(".txt"):
            # Flat: resource_X.txt  ->  label_X.json
            label_name = item.replace("resource_", "label_", 1).replace(".txt", ".json")
            json_path = os.path.join(label_dir, label_name)
            if os.path.isfile(json_path):
                pairs.append((item_path, json_path))
            else:
                print(f"  [WARN] No label for: {item}")

        elif os.path.isdir(item_path):
            # Nested: TS_*/resource_X.txt  ->  TL_*/label_X.json
            dir_name = item
            label_subdir = re.sub(
                r"^(TS|VS)_",
                lambda m: {"TS": "TL", "VS": "VL"}[m.group(1)] + "_",
                dir_name,
            )
            label_subdir_path = os.path.join(label_dir, label_subdir)

            for txt_file in sorted(os.listdir(item_path)):
                if not txt_file.endswith(".txt"):
                    continue
                txt_path = os.path.join(item_path, txt_file)
                label_name = txt_file.replace("resource_", "label_", 1).replace(".txt", ".json")
                json_path = os.path.join(label_subdir_path, label_name)
                if os.path.isfile(json_path):
                    pairs.append((txt_path, json_path))
                else:
                    print(f"  [WARN] No label for: {os.path.join(dir_name, txt_file)}")

    return pairs


def extract_session_num(json_path):
    """Extract session number from filename, e.g. label_depression_1_check_D002.json -> 1"""
    m = re.search(r"label_\w+_(\d+)_check_", os.path.basename(json_path))
    return int(m.group(1)) if m else None


def build_csv(split_dir, out_path):
    pairs = collect_sessions(split_dir)
    rows = []
    skipped = 0

    for txt_path, json_path in pairs:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            print(f"  [WARN] Empty text file: {txt_path}")
            skipped += 1
            continue

        labels = read_json_labels(json_path)
        if labels.get("anxiety") is None:
            print(f"  [WARN] Could not parse anxiety from: {json_path}")
            skipped += 1
            continue

        rows.append({
            "id":          labels.get("id"),
            "session_num": extract_session_num(json_path),
            "age":         labels.get("age"),
            "gender":      labels.get("gender"),
            "depression":  labels.get("depression"),
            "anxiety":     labels.get("anxiety"),
            "addiction":   labels.get("addiction"),
            "class":       labels.get("class"),
            "summary":     labels.get("summary"),
            "text":        text,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values(["id", "session_num"], na_position="last").reset_index(drop=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} sessions (skipped {skipped}) -> {out_path}")
    print(f"  Columns: {list(df.columns)}")
    if "class" in df.columns:
        print(f"  class distribution: {df['class'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("Building train CSV...")
    build_csv(os.path.join(base, "train"), os.path.join(base, "english_train.csv"))

    print("\nBuilding valid CSV...")
    build_csv(os.path.join(base, "valid"), os.path.join(base, "english_valid.csv"))
