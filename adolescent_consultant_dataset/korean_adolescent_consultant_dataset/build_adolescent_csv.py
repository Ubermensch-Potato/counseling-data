"""
Build korean_adolescent_train.csv and korean_adolescent_valid.csv.
Each row = one session (.txt file + .json label).

Directory layout:
  train/inputs/{category}/*.txt   <->   train/labels/{category}/*.json
  valid/inputs/{category}/*.txt   <->   valid/labels/{category}/*.json

Filename mapping: same numeric ID, same category subdir.
  e.g. inputs/응급/0008.txt  <->  labels/응급/0008.json

All fields from the JSON "info" key are collected as columns, plus:
  - category  : subdir name (e.g. "응급", "정상군", ...)
  - text       : full conversation text from the .txt file
"""

import os
import json
import pandas as pd


INFO_FIELDS = [
    "ID",
    "성별",
    "나이",
    "학년",
    "유형구분",
    "가정환경",
    "상담일자",
    "평가일시",
    "작성자(상담사)",
    "상호작용 특성(종합)",
    "긴장 수준(종합)",
    "행동 특성(종합)",
    "위기단계",
    "합계점수",
    "학대의심",
    "행동특성 점수",
    "임상가 종합소견",
]


def read_json_info(json_path):
    """Read the 'info' dict from a JSON label file. Falls back to empty dict on error."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        info = d.get("info", {})
        return {k: info.get(k) for k in INFO_FIELDS}
    except Exception as e:
        print(f"  [WARN] Could not parse JSON: {json_path} — {e}")
        return {k: None for k in INFO_FIELDS}


def collect_sessions(split_dir):
    """
    Walk split_dir/inputs/{category}/ and yield (category, txt_path, json_path) triples.
    """
    input_dir = os.path.join(split_dir, "inputs")
    label_dir = os.path.join(split_dir, "labels")
    triples = []

    for category in sorted(os.listdir(input_dir)):
        cat_input = os.path.join(input_dir, category)
        cat_label = os.path.join(label_dir, category)

        if not os.path.isdir(cat_input):
            continue
        if not os.path.isdir(cat_label):
            print(f"  [WARN] No label subdir for category: {category}")
            continue

        for fname in sorted(os.listdir(cat_input)):
            if not fname.endswith(".txt"):
                continue
            txt_path = os.path.join(cat_input, fname)
            json_name = fname.replace(".txt", ".json")
            json_path = os.path.join(cat_label, json_name)

            if os.path.isfile(json_path):
                triples.append((category, txt_path, json_path))
            else:
                print(f"  [WARN] No label for: {category}/{fname}")

    return triples


def build_csv(split_dir, out_path):
    triples = collect_sessions(split_dir)
    rows = []
    skipped = 0

    for category, txt_path, json_path in triples:
        # Read full session text
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                text = f.read().strip()
        except Exception as e:
            print(f"  [WARN] Could not read: {txt_path} — {e}")
            skipped += 1
            continue

        if not text:
            print(f"  [WARN] Empty text file: {txt_path}")
            skipped += 1
            continue

        info = read_json_info(json_path)
        row = {"category": category}
        row.update(info)
        row["text"] = text
        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by category then ID for consistency
    sort_cols = [c for c in ["category", "ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Saved {len(df)} sessions (skipped {skipped}) -> {out_path}")
    print(f"  Columns: {list(df.columns)}")
    if "위기단계" in df.columns:
        print(f"  위기단계 distribution: {df['위기단계'].value_counts().to_dict()}")
    return df


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    print("Building train CSV...")
    build_csv(
        os.path.join(base, "train"),
        os.path.join(base, "korean_adolescent_train.csv"),
    )

    print("\nBuilding valid CSV...")
    build_csv(
        os.path.join(base, "valid"),
        os.path.join(base, "korean_adolescent_valid.csv"),
    )
