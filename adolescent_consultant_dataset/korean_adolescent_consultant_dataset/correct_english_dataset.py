# conda activate TL && python "C:\Users\cwc\Desktop\cwc\TL\foundation data\adolescent_consultant_dataset\korean_adolescent_consultant_dataset\correct_english_dataset.py"
"""
Post-correction: fix first-person pronouns in Counselor lines of the
already-translated English adolescent counseling dataset.

The issue: The Korean source uses first-person pronouns (나/내/나를) in
counselor questions because the original checklist was written from the
child's perspective. The translator carried these over literally (I/me/my),
but in English the counselor must address the client in second-person (you/your).

Mirrors english_adolescent_consultant_dataset/ → english_adolescent_consultant_dataset_corrected/
  • .txt files  → pronoun-corrected
  • .json files → audio[].text updated to match corrected txt (positional)
"""

import re
import json
import ollama
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Ollama settings ───────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_CTX   = 20480
NUM_CTX_OPTIONS = {"num_ctx": OLLAMA_CTX, "num_thread": 12}
# Parallel workers — each sends independent requests to ollama concurrently.
# Set OLLAMA_NUM_PARALLEL >= NUM_WORKERS on the ollama server:
#   Windows : $env:OLLAMA_NUM_PARALLEL=4; ollama serve
#   Linux   : OLLAMA_NUM_PARALLEL=4 ollama serve
NUM_WORKERS = 4

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).parent.parent
SRC_ROOT = BASE / "english_adolescent_consultant_dataset"
DST_ROOT = BASE / "english_adolescent_consultant_dataset_corrected"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a dialogue post-editor for psychological counseling transcripts.

You will receive numbered English dialogue lines in the format:
  [N] Speaker : utterance

Your task is to fix ONLY the Counselor lines that use first-person pronouns
(I, me, my, myself, I'm, I've, I'll, I'd) when referring to the CLIENT's
experiences, feelings, body, or situation — and convert them to second-person
(you, your, yourself, you're, you've, you'll, you'd).

Rules:
  1. ONLY edit Counselor lines. Never touch Client lines.
  2. Within a Counselor line, ONLY change pronouns that refer to the CLIENT —
     i.e., the counselor is asking the client about THEIR OWN life.
     These arise from Korean checklist items originally written from the
     child's first-person viewpoint (나/내/나를).
     Examples of WRONG → CORRECT:
       "Who is the adult who takes care of me?" → "...takes care of you?"
       "Have I ever felt so troubled that I wanted to harm myself?" → "Have you ever felt...harm yourself?"
       "Has anyone hit me?" → "Has anyone hit you?"
       "When I'm feeling down, do I talk to family?" → "When you're feeling down, do you talk to family?"
  3. Do NOT change first-person pronouns that belong to the counselor
     speaking about themselves:
       "I understand." → leave as-is
       "I'm going to ask you a few questions." → leave as-is
       "I think that must be hard." → leave as-is
  4. Do NOT change anything else — wording, punctuation, line numbers,
     speaker labels, Client lines. Only fix the pronoun perspective.
  5. Output EXACTLY the same number of lines as input, each starting with [N].
     No blank lines, no commentary, no explanations.
"""


# ── Ollama helpers ────────────────────────────────────────────────────────────

def call_ollama(system: str, user: str) -> str:
    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        options=NUM_CTX_OPTIONS,
        keep_alive=-1,
        stream=False,
    )
    return response.message.content.strip()


def number_lines(lines: list[str]) -> str:
    return "\n".join(f"[{i+1}] {ln}" for i, ln in enumerate(lines))


def parse_indexed_lines(response: str, n: int) -> list[str] | None:
    tagged = []
    for raw in response.splitlines():
        m = re.match(r"^\[(\d+)\]\s*(.*)", raw.strip())
        if m:
            tagged.append(m.group(2))
    return tagged if len(tagged) == n else None


# ── Post-correction ───────────────────────────────────────────────────────────

def restore_client_lines(original: list[str], corrected: list[str]) -> list[str]:
    """Hard guard: any Client line modified by the model is silently restored."""
    result = []
    for orig, corr in zip(original, corrected):
        if orig.startswith("Client :") and orig != corr:
            tqdm.write(f"  [GUARD] Restored wrongly modified Client line: {corr!r} → {orig!r}")
            result.append(orig)
        else:
            result.append(corr)
    return result


def post_correct(lines: list[str], label: str) -> list[str]:
    """Send all lines to ollama and return the corrected version."""
    user_msg = (
        f"Fix the Counselor pronoun perspective in the following dialogue.\n"
        f"Output EXACTLY {len(lines)} lines, each starting with [N]:\n\n"
        f"{number_lines(lines)}"
    )
    raw = call_ollama(SYSTEM_PROMPT, user_msg)
    result = parse_indexed_lines(raw, len(lines))
    if result is None:
        got = sum(1 for l in raw.splitlines() if re.match(r"^\[\d+\]", l.strip()))
        tqdm.write(f"  [WARN] {label}: expected {len(lines)} lines, got {got} — keeping original")
        return lines
    return restore_client_lines(lines, result)


# ── JSON update ───────────────────────────────────────────────────────────────

def strip_speaker_prefix(line: str) -> str:
    m = re.match(r"^(?:Counselor|Client)\s*:\s*(.*)", line, re.IGNORECASE)
    return m.group(1).strip() if m else line.strip()


def update_json(src_json: Path, original_lines: list[str], corrected_lines: list[str]) -> dict:
    """Update audio[].text entries positionally where lines changed."""
    data = json.loads(src_json.read_text(encoding="utf-8"))

    corrections = {
        i: strip_speaker_prefix(corr)
        for i, (orig, corr) in enumerate(zip(original_lines, corrected_lines))
        if orig != corr
    }
    if not corrections:
        return data

    counter = [0]

    def walk(node):
        if not isinstance(node, dict):
            return
        for section in node.get("list", []):
            if not isinstance(section, dict):
                continue
            for item in section.get("list", []):
                if not isinstance(item, dict):
                    continue
                for audio in item.get("audio", []):
                    if not isinstance(audio, dict):
                        continue
                    idx = counter[0]
                    if idx in corrections:
                        audio["text"] = corrections[idx]
                    counter[0] += 1

    walk(data)

    if counter[0] != len(original_lines):
        tqdm.write(f"  [WARN] {src_json.name}: audio entries ({counter[0]}) != txt lines ({len(original_lines)})")

    return data


# ── File processor ────────────────────────────────────────────────────────────

def process_file(src_txt: Path, skip_existing: bool) -> str:
    """Correct one txt+json pair. Returns a short status string."""
    rel      = src_txt.relative_to(SRC_ROOT)                              # split/inputs/cat/NNNN.txt
    dst_txt  = DST_ROOT / rel
    src_json = SRC_ROOT / rel.parts[0] / "labels" / Path(*rel.parts[2:]).with_suffix(".json")
    dst_json = DST_ROOT / rel.parts[0] / "labels" / Path(*rel.parts[2:]).with_suffix(".json")
    label    = str(rel)

    if skip_existing and dst_txt.exists() and dst_json.exists():
        return "SKIP"

    try:
        original = src_txt.read_text(encoding="utf-8").strip().splitlines()
        corrected = post_correct(original, label)
        n_changed = sum(o != c for o, c in zip(original, corrected))

        dst_txt.parent.mkdir(parents=True, exist_ok=True)
        dst_txt.write_text("\n".join(corrected), encoding="utf-8")

        if src_json.exists():
            updated = update_json(src_json, original, corrected)
            dst_json.parent.mkdir(parents=True, exist_ok=True)
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(updated, f, ensure_ascii=False, indent=4)

        return f"OK ({n_changed} lines fixed)"

    except Exception as e:
        tqdm.write(f"  [ERROR] {label}: {e}")
        return "ERROR"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    all_files = sorted(SRC_ROOT.rglob("*.txt"))
    skip_existing = True

    if skip_existing:
        todo = [f for f in all_files
                if not (DST_ROOT / f.relative_to(SRC_ROOT)).exists()]
    else:
        todo = all_files

    print(f"Source : {SRC_ROOT}")
    print(f"Output : {DST_ROOT}")
    print(f"Files  : {len(todo)} to process (skipping {len(all_files) - len(todo)} already done) — {NUM_WORKERS} parallel workers\n")

    futures = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for src_txt in todo:
            fut = pool.submit(process_file, src_txt, skip_existing)
            futures[fut] = src_txt.name

        with tqdm(total=len(todo), unit="file") as pbar:
            for fut in as_completed(futures):
                name = futures[fut]
                status = fut.result()
                pbar.set_postfix_str(f"{name} → {status}"[:60])
                pbar.update(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
