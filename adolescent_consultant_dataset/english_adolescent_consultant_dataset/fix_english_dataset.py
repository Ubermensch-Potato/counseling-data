"""
fix_english_dataset.py

Deterministic (no LLM) post-fix for the english_adolescent_consultant_dataset_corrected
directory. Reads from train/ and valid/ (read-only), writes fixed copies to
post-train/ and post-valid/.

Five error classes fixed:
  1. JSON audio[].type has Q/A swapped        → re-synced from TXT
  2. JSON audio[].text uses wrong pronoun      → re-synced from TXT
  3. JSON has blank audio entries              → removed in-place before alignment
  4. TXT has extra trailing lines              → trimmed to match JSON audio count
  5. TXT/JSON have consecutive duplicate lines → second occurrence removed from both

TXT is authoritative for speaker assignment and text content.
JSON is authoritative for structure, timestamps, scores.
"""

import json
import sys
import io
from pathlib import Path
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

BASE       = Path(__file__).parent
SPLITS     = ["train", "valid"]
CATEGORIES = ["정상군", "관찰필요", "상담필요", "응급", "학대의심"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_txt(path: Path) -> list[tuple[str, str]]:
    """Return [(type, text), ...] where type is 'Q' or 'A' (or '?' if unknown)."""
    lines = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("Counselor :"):
                lines.append(("Q", line[len("Counselor :"):].strip()))
            elif line.startswith("Client :"):
                lines.append(("A", line[len("Client :"):].strip()))
            else:
                lines.append(("?", line))
    return lines


def collect_audio_refs(label: dict) -> list[dict]:
    """Return list of audio entry dicts (by reference) in traversal order."""
    refs = []
    for domain in label.get("list", []):
        if not isinstance(domain, dict):
            continue
        for item in domain.get("list", []):
            if not isinstance(item, dict):
                continue
            for entry in item.get("audio", []):
                if isinstance(entry, dict):
                    refs.append(entry)
    return refs


def remove_blank_audio(label: dict) -> int:
    """Remove audio entries where text.strip() == '' in-place. Returns count removed."""
    removed = 0
    for domain in label.get("list", []):
        if not isinstance(domain, dict):
            continue
        for item in domain.get("list", []):
            if not isinstance(item, dict):
                continue
            audio_list = item.get("audio", [])
            new_list = [e for e in audio_list if isinstance(e, dict) and e.get("text", "").strip() != ""]
            n_removed = len(audio_list) - len(new_list)
            if n_removed:
                item["audio"] = new_list
                removed += n_removed
    return removed


def remove_duplicate_audio(label: dict, dup_indices: set) -> int:
    """Remove audio entries at the given 0-based traversal indices in-place.
    Returns count removed."""
    removed = 0
    counter = 0
    for domain in label.get("list", []):
        if not isinstance(domain, dict):
            continue
        for item in domain.get("list", []):
            if not isinstance(item, dict):
                continue
            audio_list = item.get("audio", [])
            new_list = []
            for entry in audio_list:
                if counter not in dup_indices:
                    new_list.append(entry)
                else:
                    removed += 1
                counter += 1
            item["audio"] = new_list
    return removed


def format_txt_line(type_: str, text: str) -> str:
    prefix = "Counselor" if type_ == "Q" else "Client"
    return f"{prefix} : {text}"


# ── Core processor ────────────────────────────────────────────────────────────

def process_file(
    txt_path: Path,
    json_path: Path,
    out_txt_path: Path,
    out_json_path: Path,
    stats: dict,
) -> str:
    """
    Process one (txt, json) pair.
    Writes fixed outputs and updates stats dict.
    Returns a status string.
    """
    # Load sources
    txt_lines = load_txt(txt_path)

    with open(json_path, encoding="utf-8") as f:
        label = json.load(f)
    label_copy = deepcopy(label)

    # ── Step 1: Remove blank JSON audio entries ────────────────────────────────
    blank_removed = remove_blank_audio(label_copy)
    if blank_removed:
        stats["blank_files"].add(str(json_path))
        stats["blank_entries"] += blank_removed

    # ── Step 2: Remove extra trailing TXT lines ────────────────────────────────
    json_audio = collect_audio_refs(label_copy)
    diff = len(txt_lines) - len(json_audio)
    extra_removed = 0
    if diff > 0:
        txt_lines = txt_lines[:-diff]
        extra_removed = diff
        stats["extra_files"].add(str(txt_path))
        stats["extra_lines"] += extra_removed

    # ── Step 3: Validate alignment ─────────────────────────────────────────────
    json_audio = collect_audio_refs(label_copy)   # re-collect after blank removal
    if len(txt_lines) != len(json_audio):
        stats["skipped"].append(
            f"{txt_path.parent.name}/{txt_path.name}: "
            f"txt={len(txt_lines)} json={len(json_audio)} (after removing {blank_removed} blank, {extra_removed} extra)"
        )
        return f"SKIP (txt={len(txt_lines)} json={len(json_audio)})"

    # ── Step 4: Re-sync JSON audio entries from TXT ────────────────────────────
    type_fixed = 0
    text_fixed = 0
    for i, entry in enumerate(json_audio):
        new_type, new_text = txt_lines[i]
        # Only override type when TXT clearly identifies the speaker ("Q"/"A").
        # If TXT line had no recognised prefix ("?"), keep the original JSON type
        # to avoid introducing a "?" that would mismatch when re-read from TXT.
        if new_type in ("Q", "A"):
            if entry.get("type") != new_type:
                entry["type"] = new_type
                type_fixed += 1
        # Only override text when TXT provides non-empty content.
        # Empty text from a malformed TXT line (e.g. "Client :  ") should not
        # blank out the JSON entry.
        if new_text != "":
            if entry.get("text", "").strip() != new_text:
                entry["text"] = new_text
                text_fixed += 1

    if type_fixed:
        stats["type_files"].add(str(json_path))
        stats["type_entries"] += type_fixed
    if text_fixed:
        stats["text_files"].add(str(json_path))
        stats["text_entries"] += text_fixed

    # ── Step 5: Remove consecutive duplicate lines ─────────────────────────────
    # After re-sync, TXT and JSON are aligned. Walk forward: if line i is
    # identical (same type AND same text) to line i-1, it is a duplicate —
    # remove it from both TXT and JSON.
    # Re-derive the current (type, text) pairs from the updated json_audio so
    # that "?" entries use their preserved JSON type.
    synced = []
    for entry in json_audio:
        t = entry.get("type", "A")
        if t not in ("Q", "A"):
            t = "A"
        synced.append((t, entry.get("text", "").strip()))

    dup_indices = set()
    for i in range(1, len(synced)):
        if synced[i] == synced[i - 1]:
            dup_indices.add(i)

    dup_removed = 0
    if dup_indices:
        dup_removed = remove_duplicate_audio(label_copy, dup_indices)
        # Re-collect refs after removal so the write step uses the pruned list
        json_audio = collect_audio_refs(label_copy)
        stats["dup_files"].add(str(json_path))
        stats["dup_entries"] += dup_removed

    # ── Step 6: Write outputs ──────────────────────────────────────────────────
    out_txt_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)

    # Write TXT using the final JSON entry types (so "?" TXT lines use the
    # preserved JSON type and the output TXT/JSON stay in sync).
    out_lines = []
    for i, entry in enumerate(json_audio):
        final_type = entry.get("type", "A")
        if final_type not in ("Q", "A"):
            final_type = "A"
        final_text = entry.get("text", "").strip()
        out_lines.append(format_txt_line(final_type, final_text))
    txt_content = "\n".join(out_lines)
    out_txt_path.write_text(txt_content, encoding="utf-8")

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(label_copy, f, ensure_ascii=False, indent=4)

    parts = []
    if blank_removed:
        parts.append(f"blank={blank_removed}")
    if extra_removed:
        parts.append(f"extra={extra_removed}")
    if type_fixed:
        parts.append(f"type={type_fixed}")
    if text_fixed:
        parts.append(f"text={text_fixed}")
    if dup_removed:
        parts.append(f"dup={dup_removed}")
    return f"OK ({', '.join(parts)})" if parts else "OK (no changes)"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    stats = {
        "blank_files":  set(),
        "blank_entries": 0,
        "extra_files":  set(),
        "extra_lines":  0,
        "type_files":   set(),
        "type_entries": 0,
        "text_files":   set(),
        "text_entries": 0,
        "dup_files":    set(),
        "dup_entries":  0,
        "skipped":      [],
        "processed":    0,
        "errors":       [],
    }

    for split in SPLITS:
        out_split = f"post-{split}"
        for cat in CATEGORIES:
            in_input_dir  = BASE / split  / "inputs" / cat
            in_label_dir  = BASE / split  / "labels" / cat
            out_input_dir = BASE / out_split / "inputs" / cat
            out_label_dir = BASE / out_split / "labels" / cat

            if not in_input_dir.exists():
                continue

            for txt_path in sorted(in_input_dir.glob("*.txt")):
                case_id   = txt_path.stem
                json_path = in_label_dir / f"{case_id}.json"

                if not json_path.exists():
                    stats["errors"].append(f"Missing JSON: {json_path}")
                    continue

                out_txt_path  = out_input_dir / txt_path.name
                out_json_path = out_label_dir / f"{case_id}.json"

                try:
                    status = process_file(
                        txt_path, json_path,
                        out_txt_path, out_json_path,
                        stats,
                    )
                    stats["processed"] += 1
                    if not status.startswith("OK (no"):
                        print(f"  [{split}/{cat}/{case_id}] {status}")
                except Exception as e:
                    stats["errors"].append(f"{split}/{cat}/{case_id}: {e}")
                    print(f"  [ERROR] {split}/{cat}/{case_id}: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Files processed                 : {stats['processed']}")
    print(f"Blank JSON entries removed      : {len(stats['blank_files'])} files, {stats['blank_entries']} entries")
    print(f"Extra TXT lines removed         : {len(stats['extra_files'])} files, {stats['extra_lines']} lines")
    print(f"Type fields fixed in JSON       : {len(stats['type_files'])} files, {stats['type_entries']} entries")
    print(f"Text fields fixed in JSON       : {len(stats['text_files'])} files, {stats['text_entries']} entries")
    print(f"Consecutive duplicates removed  : {len(stats['dup_files'])} files, {stats['dup_entries']} entries")
    print(f"Skipped (count mismatch)        : {len(stats['skipped'])} files")
    if stats["skipped"]:
        for s in stats["skipped"]:
            print(f"  SKIP: {s}")
    if stats["errors"]:
        print(f"Errors                          : {len(stats['errors'])}")
        for e in stats["errors"]:
            print(f"  ERROR: {e}")
    print("=" * 70)
    print("Done. Run sample_check.py against post-train/ to verify error counts → 0.")


if __name__ == "__main__":
    main()
