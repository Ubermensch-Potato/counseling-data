# conda activate TL && python "C:\Users\cwc\Desktop\cwc\TL\foundation data\counseling_dataset\korean_counseling_dataset\translate_to_english.py"
"""
Translate Korean counseling .txt files to English using ollama (qwen2.5:14b).

Mirrors korean_counseling_dataset/ → english_counseling_dataset/ (sibling directory).
  • .txt files  → translated to English (speaker labels in English)
  • .json files → translated: paragraph_text, paragraph_speaker, summary updated;
                  all numeric/id fields kept as-is

Long files are split into indexed chunks ([N] prefix per line) so the model
returns [N]-tagged output that can be parsed by index regardless of blank lines
or formatting differences.  The last OVERLAP_LINES of already-translated English
are fed as leading context to the next chunk for continuity.
"""

import re
import json
import shutil
import time
import threading
import unicodedata
import ollama
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# python "D:\cwc\TL\foundation data\korean_counseling_dataset\translate_to_english.py"

# ── Error log (set in __main__, written thread-safely) ───────────────────────
_error_log_path: Path | None = None
_error_log_lock = threading.Lock()

def _write_error_log(path: str) -> None:
    if _error_log_path is None:
        return
    with _error_log_lock:
        if _error_log_path.exists():
            with open(_error_log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"error_files": []}
        data["error_files"].append(path)
        with open(_error_log_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
# ── Ollama settings ──────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen3:14b"
OLLAMA_CTX   = 32768       # num_ctx passed to the model (tokens): 32768, 16384 8192
# Parallel workers — each sends independent requests to ollama concurrently.
# Set OLLAMA_NUM_PARALLEL >= NUM_WORKERS on the ollama server:
#   Windows : $env:OLLAMA_NUM_PARALLEL=4; ollama serve
#   Linux   : OLLAMA_NUM_PARALLEL=4 ollama serve
NUM_WORKERS  = 2

# Reserve this many tokens for the model's output; the rest is the input budget.
RESPONSE_RESERVE = 2048    # tokens kept for the generated translation
# Korean text is token-dense: ~1.5 chars per token is a conservative estimate.
CHARS_PER_TOKEN  = 1.5

# ── Chunking settings ────────────────────────────────────────────────────────
CHUNK_LINES   = 80         # dialogue turns per chunk to translate at once
OVERLAP_LINES = 20         # tail lines of previous chunk fed as context (not re-output)

# ── Non-English character detection ──────────────────────────────────────────
_NON_ASCII_RE = re.compile(r'[^\x00-\x7F]')


def _is_foreign_script(ch: str) -> bool:
    """Return True if ch is a non-ASCII letter/digit from a non-Latin script.

    Allows accented Latin letters used in English loanwords (é, ü, ñ, etc.)
    by checking that the Unicode name does NOT start with 'LATIN'.
    """
    return (
        unicodedata.category(ch)[0] in ('L', 'N')
        and not unicodedata.name(ch, '').startswith('LATIN')
    )


def _has_non_english(line: str) -> bool:
    """Return True if line contains non-ASCII, non-Latin letters or digits."""
    return any(_is_foreign_script(ch) for ch in _NON_ASCII_RE.findall(line))


def _get_non_english_chars(line: str) -> list[str]:
    """Return the specific foreign-script characters found in line."""
    return [ch for ch in _NON_ASCII_RE.findall(line) if _is_foreign_script(ch)]


# ── Speaker label mapping ─────────────────────────────────────────────────────
SPEAKER_MAP = {
    "상담사": "Counselor",
    "내담자": "Client",
}

SYSTEM_PROMPT = (
    "You are a professional translator specializing in psychological counseling transcripts. "
    "The user will give you numbered Korean dialogue lines in the format [N] Speaker : utterance. "
    "Translate each line into English and output it as [N] Speaker : utterance — one output line per input line. "
    "Rules:\n"
    "  1. Replace '상담사' with 'Counselor' and '내담자' with 'Client' as speaker labels.\n"
    "  2. Keep placeholder tokens (e.g. @COUNSELOR, @TIME, @PLACE, @HOSPITAL) exactly as-is — do NOT translate them.\n"
    "  3. Output ONLY the numbered translated lines — no blank lines, no explanations, no commentary.\n"
    "  4. Every input [N] must have exactly one output [N]. Do not skip, merge, or split indices.\n"
    "  5. Counselor lines are questions directed AT the Client. "
    "Korean counselor questions sometimes use first-person perspective (나/내가/나를/내/저) because the original Korean is phrased from the client's viewpoint. "
    "Always translate these into second-person English (you/your/yourself), NOT first-person (I/me/my/myself). "
    "Example: '내가 죽고 싶다는 생각을 해본 적 있나요?' → 'Have you ever thought about wanting to die?' (NOT 'Have I ever...'). "
    "Example: '내 몸을 동의 없이 만진 사람이 있나요?' → 'Has anyone touched your body without your consent?' (NOT '...my body...').\n"
    "  6. Korean counselors sometimes refer to the Client in the third person using terms like '우리 친구' (our friend) or '친구' (friend). "
    "Always translate these as direct second-person address (you/your). "
    "Example: '우리 친구는 요즘 뭐가 제일 걱정돼요?' → 'What are you most worried about these days?' (NOT 'What is our friend most worried about?').\n"
    "  7. Each Counselor question must match what the Client answers. "
    "If the Korean question asks about the Client's own feelings or experiences (e.g. 친구가/네가 뭘 좋아해?), translate it as a direct question to the Client (you), "
    "not as a question about a third party. "
    "Example: '친구는 뭘 할 때 행복해요?' → 'What makes you happy?' (NOT 'What makes your friends happy?').\n"
    "  8. Translate every utterance literally and accurately, including short, colloquial, or dialectal expressions. "
    "Do not substitute a contextually plausible phrase when the actual meaning differs — short lines must convey exactly what was said, not what seems natural from surrounding context. "
    "If a word is colloquial or dialectal, find its correct meaning rather than guessing from context.\n"
    "If 'Previous context' lines are given, use them only to maintain continuity — do NOT retranslate them."
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a professional translator specializing in psychological counseling reports. "
    "Translate the following Korean counseling session summary into natural English. "
    "Output only the translated text — no explanations, no commentary."
)


# ─────────────────────────────────────────────────────────────────────────────
# Token-budget guard
# ─────────────────────────────────────────────────────────────────────────────

class ContextTooLongError(ValueError):
    pass


def estimate_tokens(text: str) -> int:
    """Rough token estimate: Korean chars are token-dense (~1.5 chars/token)."""
    return int(len(text) / CHARS_PER_TOKEN)


def check_context_budget(system_text: str, user_text: str) -> None:
    """Raise ContextTooLongError if the prompt exceeds the available input budget."""
    prompt_tokens = estimate_tokens(system_text) + estimate_tokens(user_text)
    budget = OLLAMA_CTX - RESPONSE_RESERVE
    if prompt_tokens > budget:
        raise ContextTooLongError(
            f"Prompt too long: ~{prompt_tokens} estimated tokens > "
            f"{budget} input budget (OLLAMA_CTX={OLLAMA_CTX} - RESPONSE_RESERVE={RESPONSE_RESERVE}). "
            f"Reduce CHUNK_LINES or increase OLLAMA_CTX."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama helpers
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(system_prompt: str, user_text: str, retries: int = 3) -> str:
    check_context_budget(system_prompt, user_text)

    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_text},
                ],
                options={
                    "num_ctx": OLLAMA_CTX,
                    "num_thread": 12,
                },
                keep_alive=-1,
                stream=False,
            )
            return response.message.content.strip()
        except Exception as e:
            print(f"    [WARN] Ollama attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
    raise RuntimeError(f"Ollama failed after {retries} attempts.")


def translate_summary(korean_summary: str) -> str:
    """Translate the top-level summary field via a dedicated single call."""
    return call_ollama(SUMMARY_SYSTEM_PROMPT, korean_summary)


# ─────────────────────────────────────────────────────────────────────────────
# Indexed-line helpers
# ─────────────────────────────────────────────────────────────────────────────

def number_lines(lines: list[str], start_idx: int = 1) -> str:
    """Prefix each line with a 1-based global index: [N] line."""
    return "\n".join(f"[{start_idx + i}] {ln}" for i, ln in enumerate(lines))


def parse_indexed_lines(response: str, n: int) -> list[str] | None:
    """
    Extract translated lines by positional order of [N] markers.

    The model's actual [N] numbers are ignored — only their order matters.
    Returns a list of exactly `n` strings, or None if the count doesn't match
    (caller should retry).
    """
    tagged_lines = []
    for raw_line in response.splitlines():
        m = re.match(r"^\[(\d+)\]\s*(.*)", raw_line.strip())
        if m:
            tagged_lines.append(m.group(2))

    if len(tagged_lines) != n:
        return None

    return tagged_lines


# ─────────────────────────────────────────────────────────────────────────────
# Translation logic (chunked with indexed lines)
# ─────────────────────────────────────────────────────────────────────────────

MAX_LINE_RETRIES = 20   # retries before splitting on line-count mismatch


def _build_user_msg(chunk: list[str], global_start: int,
                    ctx_lines: list[str], ctx_global_start: int) -> str:
    n = len(chunk)
    numbered_chunk = number_lines(chunk, start_idx=global_start)
    if ctx_lines:
        ctx_numbered = number_lines(ctx_lines, start_idx=ctx_global_start)
        return (
            f"Previous context (already translated English, for continuity only):\n"
            f"{ctx_numbered}\n\n"
            f"Translate the following continuation. "
            f"Output EXACTLY {n} lines, each starting with [N]:\n\n"
            f"{numbered_chunk}"
        )
    return (
        f"Translate the following dialogue. "
        f"Output EXACTLY {n} lines, each starting with [N]:\n\n"
        f"{numbered_chunk}"
    )


def _split_and_recurse(chunk: list[str], global_start: int,
                       ctx_lines: list[str], ctx_global_start: int,
                       pbar: tqdm, depth: int) -> list[str]:
    n = len(chunk)
    half = n // 2
    pbar.total += 1
    pbar.refresh()

    first = _translate_chunk(chunk[:half], global_start, [], 0, pbar, depth + 1)

    new_ctx_start = global_start + max(0, half - OVERLAP_LINES)
    new_ctx = first[max(0, half - OVERLAP_LINES):]
    second = _translate_chunk(chunk[half:], global_start + half,
                              new_ctx, new_ctx_start, pbar, depth + 1)
    return first + second


def _translate_chunk(
    chunk: list[str],
    global_start: int,          # 1-based index of chunk[0]
    ctx_lines: list[str],
    ctx_global_start: int,      # 1-based index of ctx_lines[0]
    pbar: tqdm,
    depth: int = 0,
) -> list[str]:
    """
    Translate `chunk`.

    - ContextTooLongError → recursively halve the chunk.
    - Wrong line count    → retry up to MAX_LINE_RETRIES with an explicit
                            count hint injected; halve if still wrong.
    - n == 1 and unsolvable → skip with error log.
    """
    n = len(chunk)
    chunk_id = f"lines {global_start}-{global_start+n-1}"
    user_msg = _build_user_msg(chunk, global_start, ctx_lines, ctx_global_start)

    # ── ContextTooLongError handling ─────────────────────────────────────────
    try:
        check_context_budget(SYSTEM_PROMPT, user_msg)
    except ContextTooLongError as e:
        if n == 1:
            tqdm.write(f"    [ERROR] {chunk_id} exceeds context limit even alone — skipping.\n"
                       f"            {e}")
            pbar.update(1)
            return [""]
        tqdm.write(f"    [WARN] ContextTooLongError ({chunk_id}, {n} lines) — splitting into "
                   f"{n//2} + {n - n//2}.")
        return _split_and_recurse(chunk, global_start, ctx_lines, ctx_global_start, pbar, depth)

    # ── Translation + line-count retry loop ──────────────────────────────────
    for attempt in range(1, MAX_LINE_RETRIES + 1):
        raw = call_ollama(SYSTEM_PROMPT, user_msg)
        result = parse_indexed_lines(raw, n)

        if result is not None:
            # ── Check for non-English characters in the translated output ─────
            bad_indices = [i for i, ln in enumerate(result) if _has_non_english(ln)]
            if not bad_indices:
                pbar.update(1)
                return result

            bad_global = [global_start + i for i in bad_indices]
            bad_lines_info = ", ".join(
                f"line {global_start + i}: \"{result[i]}\""
                for i in bad_indices
            )
            tqdm.write(
                f"    [WARN] {chunk_id}: translated output still contains non-English "
                f"characters at global line(s) {bad_global} — {bad_lines_info} "
                f"(attempt {attempt}/{MAX_LINE_RETRIES}). Retrying with language hint…"
            )
            user_msg = (
                f"WARNING: your previous response contained non-English (e.g. Korean) "
                f"characters at output line(s) {[i + 1 for i in bad_indices]}. "
                f"You MUST translate ALL text fully into English — "
                f"no Korean or other foreign-script characters are allowed in the output.\n\n"
            ) + _build_user_msg(chunk, global_start, ctx_lines, ctx_global_start)
            continue

        got = sum(1 for ln in raw.splitlines()
                  if re.match(r"^\[(\d+)\]\s*", ln.strip()))
        tqdm.write(
            f"    [WARN] {chunk_id}: expected {n} lines, got {got} "
            f"(attempt {attempt}/{MAX_LINE_RETRIES}). Retrying with count hint…"
        )
        # Inject a stronger count reminder for the next attempt
        user_msg = (
            f"IMPORTANT: your previous response had {got} lines but {n} are required.\n"
            f"Output EXACTLY {n} lines — one [N] line per input line, no more, no less.\n\n"
        ) + _build_user_msg(chunk, global_start, ctx_lines, ctx_global_start)

    # Retries exhausted — split if possible, otherwise give up
    if n < 10:
        raise RuntimeError(
            f"{chunk_id}: chunk size {n} < 10 after splitting — line count never matched."
        )

    tqdm.write(f"    [WARN] {chunk_id}: line count still wrong after {MAX_LINE_RETRIES} retries "
               f"— splitting into {n//2} + {n - n//2}.")
    return _split_and_recurse(chunk, global_start, ctx_lines, ctx_global_start, pbar, depth)


def translate_lines(korean_lines: list[str]) -> list[str]:
    """
    Translate a list of Korean dialogue lines into English.
    Returns a list of the same length with English translations.

    Every line is tagged [N] in the prompt so the model returns [N] translations.
    _translate_chunk() handles ContextTooLongError by recursively halving.
    """
    total = len(korean_lines)
    all_translated = [""] * total
    n_chunks = (total + CHUNK_LINES - 1) // CHUNK_LINES

    start = 0
    with tqdm(total=n_chunks, desc="    chunks", unit="chunk", leave=True,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}") as pbar:
        while start < total:
            end = min(start + CHUNK_LINES, total)
            chunk = korean_lines[start:end]

            pbar.set_postfix(lines=f"{start+1}-{end}/{total}")

            ctx_start = max(0, start - OVERLAP_LINES)
            ctx_lines = all_translated[ctx_start:start] if start > 0 else []

            translated_chunk = _translate_chunk(
                chunk, start + 1,
                ctx_lines, ctx_start + 1,
                pbar,
            )

            for i, text in zip(range(start, end), translated_chunk):
                all_translated[i] = text

            start = end

    return all_translated


# ─────────────────────────────────────────────────────────────────────────────
# JSON label translation
# ─────────────────────────────────────────────────────────────────────────────

def strip_speaker_prefix(translated_line: str) -> str:
    """
    'Counselor : Some utterance.' → 'Some utterance.'
    Falls back to the full line if no 'Speaker : ' pattern is found.
    """
    m = re.match(r"^(?:Counselor|Client)\s*:\s*(.*)", translated_line, re.IGNORECASE)
    return m.group(1).strip() if m else translated_line.strip()


def _fix_line_string_value(line: str) -> str:
    """
    Fix a single JSON line whose string value is malformed.

    Handles three cases:
      1. Unquoted value:         "key": raw text,      →  "key": "raw text",
      2. CSV-style inner quotes: "key": "a ""b"" c",   →  "key": "a \"b\" c",
      3. Bare inner quotes:      "key": "a "b" c",     →  "key": "a \"b\" c",

    Non-string values (numbers, booleans, null, objects, arrays) are left as-is.
    Lines that already parse correctly are returned unchanged.
    """
    # Must match:  (optional indent) "key" : value (optional trailing comma/space)
    m = re.match(r'^(\s*"(?:[^"\\]|\\.)*"\s*:\s*)(.+)$', line)
    if not m:
        return line

    prefix = m.group(1)
    rest = m.group(2).rstrip()

    # Separate trailing comma (if any)
    if rest.endswith(','):
        raw = rest[:-1].rstrip()
        trailing = ','
    else:
        raw = rest
        trailing = ''

    # If the raw value parses cleanly as JSON, nothing to do
    try:
        json.loads(raw)
        return line
    except json.JSONDecodeError:
        pass

    # Leave non-string scalars and containers alone
    if raw in ('true', 'false', 'null') or raw[:1] in ('{', '['):
        return line
    try:
        float(raw)
        return line
    except ValueError:
        pass

    # It's a broken string — strip outer quotes if present
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        inner = raw[1:-1]
    elif raw.startswith('"'):
        inner = raw[1:]
    else:
        inner = raw

    # Unescape any already-escaped quotes so we start from a clean slate
    inner = inner.replace('\\"', '"')

    # Replace CSV-style doubled quotes ("") with a placeholder, then escape all "
    _PH = '\x00'
    inner = inner.replace('""', _PH)
    inner = inner.replace('"', '\\"')
    inner = inner.replace(_PH, '\\"')

    return prefix + '"' + inner + '"' + trailing + rest[len(raw) + len(trailing):]


def sanitize_json_text(text: str) -> str:
    """
    Make JSON text spec-compliant before parsing.

    Fixes applied line by line:
      • Bare numeric literals: Infinity, -Infinity, NaN  →  quoted strings
      • Unquoted string values, CSV-style ("") inner quotes, bare inner quotes
    """
    # Fix bare numeric literals first
    text = re.sub(
        r'(?<=[:\[,])\s*(-?Infinity|NaN)\s*(?=[,\]\}])',
        lambda m: f' "{m.group(1).strip()}"',
        text,
    )
    # Fix malformed string values line by line
    lines = text.split('\n')
    fixed = []
    for line in lines:
        try:
            fixed.append(_fix_line_string_value(line))
        except Exception:
            fixed.append(line)  # never crash — fall back to original
    return '\n'.join(fixed)


def build_english_json(src_json_path: Path, translated_lines: list[str]) -> dict:
    """
    Load a Korean label JSON and produce its English counterpart:
      • paragraph_speaker  — mapped via SPEAKER_MAP
      • paragraph_text     — taken from translated_lines[paragraph["index"]]
                             with the 'Speaker : ' prefix stripped
      • summary            — translated via a separate ollama call
      • all other fields   — copied as-is
    Bare Infinity / -Infinity / NaN values (invalid JSON) are converted to
    their string equivalents before parsing.
    """
    raw = src_json_path.read_text(encoding="utf-8")
    sanitized = sanitize_json_text(raw)
    if sanitized != raw:
        print(f"    [INFO] Sanitized JSON-illegal literals in {src_json_path.name}")
    data = json.loads(sanitized)

    # Translate summary
    if data.get("summary"):
        data["summary"] = translate_summary(data["summary"])

    # Update each paragraph entry
    for para in data.get("paragraph", []):
        idx = para.get("index", None)

        # Speaker label
        kor_speaker = para.get("paragraph_speaker", "")
        para["paragraph_speaker"] = SPEAKER_MAP.get(kor_speaker, kor_speaker)

        # Utterance text — sourced from the already-translated lines
        if idx is not None and idx < len(translated_lines):
            para["paragraph_text"] = strip_speaker_prefix(translated_lines[idx])
        else:
            print(f"    [WARN] paragraph index {idx} out of range ({len(translated_lines)} lines)")

    return data


# ─────────────────────────────────────────────────────────────────────────────
# File / directory helpers
# ─────────────────────────────────────────────────────────────────────────────

def mirror_path(src_path: Path, src_root: Path, dst_root: Path) -> Path:
    return dst_root / src_path.relative_to(src_root)


def find_json_for_txt(txt_path: Path, src_inputs: Path, src_labels: Path) -> Path | None:
    """
    Derive the .json label path from a .txt input path using the same
    naming convention as build_korean_csv.py:
      flat:   inputs/resource_X.txt       → labels/label_X.json
      nested: inputs/TS_xxx/resource_X.txt → labels/TL_xxx/label_X.json
    """
    rel = txt_path.relative_to(src_inputs)
    parts = rel.parts  # e.g. ('resource_X.txt',) or ('TS_xxx', 'resource_X.txt')

    label_name = parts[-1].replace("resource_", "label_", 1).replace(".txt", ".json")

    if len(parts) == 1:
        return src_labels / label_name
    else:
        subdir = parts[0]
        label_subdir = re.sub(
            r"^(TS|VS)_",
            lambda m: {"TS": "TL", "VS": "VL"}[m.group(1)] + "_",
            subdir,
        )
        return src_labels / label_subdir / label_name


def process_file(src_txt: Path, src_inputs: Path, src_labels: Path,
                 dst_inputs: Path, dst_labels: Path,
                 skip_existing: bool, label: str) -> str:
    """Translate one .txt file and its paired .json label. Returns a status string."""
    dst_txt  = mirror_path(src_txt, src_inputs, dst_inputs)
    src_json = find_json_for_txt(src_txt, src_inputs, src_labels)
    dst_json = mirror_path(src_json, src_labels, dst_labels) if src_json else None

    dst_txt.parent.mkdir(parents=True, exist_ok=True)
    if dst_json:
        dst_json.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and dst_txt.exists() and (dst_json is None or dst_json.exists()):
        return f"SKIP  {label}"

    try:
        korean_text = src_txt.read_text(encoding="utf-8").strip()
        if not korean_text:
            return f"EMPTY {label}"

        korean_lines = korean_text.splitlines()

        # ── Translate dialogue lines ──────────────────────────────────────────
        tqdm.write(f"\n  → {label}")
        translated_lines = translate_lines(korean_lines)
        dst_txt.write_text("\n".join(translated_lines), encoding="utf-8")

        # ── Translate JSON label ──────────────────────────────────────────────
        if src_json and src_json.exists() and dst_json:
            if len(translated_lines) != len(korean_lines):
                tqdm.write(f"    [WARN] {label}: line count mismatch "
                           f"{len(translated_lines)} vs {len(korean_lines)}")
            english_json = build_english_json(src_json, translated_lines)
            with open(dst_json, "w", encoding="utf-8") as f:
                json.dump(english_json, f, ensure_ascii=False, indent=4)
        elif src_json and not src_json.exists():
            tqdm.write(f"    [WARN] No JSON label found: {src_json}")

        return f"OK    {label}"

    except Exception as e:
        tqdm.write(f"    [ERROR] {label}: {e}")
        traceback.print_exc()
        _write_error_log(label)
        return f"ERROR {label}"


def process_split(src_split: Path, dst_split: Path, skip_existing: bool = True):
    """Translate .txt inputs and their paired .json labels for one split."""
    src_inputs = src_split / "inputs"
    src_labels = src_split / "labels"
    dst_inputs = dst_split / "inputs"
    dst_labels = dst_split / "labels"

    all_txt_files = sorted(src_inputs.rglob("*.txt"))
    print(f"  {len(all_txt_files)} files total — {NUM_WORKERS} parallel workers")

    if skip_existing:
        def _is_done(src_txt: Path) -> bool:
            src_json = find_json_for_txt(src_txt, src_inputs, src_labels)
            dst_txt  = mirror_path(src_txt, src_inputs, dst_inputs)
            dst_json = mirror_path(src_json, src_labels, dst_labels) if src_json else None
            return dst_txt.exists() and (dst_json is None or dst_json.exists())
        txt_files = [f for f in all_txt_files if not _is_done(f)]
    else:
        txt_files = all_txt_files

    total = len(txt_files)
    print(f"  {total} files to process (skipping {len(all_txt_files) - total} already done)")

    futures = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        for src_txt in txt_files:
            label = str(src_txt.relative_to(src_split.parent))
            fut = pool.submit(
                process_file,
                src_txt, src_inputs, src_labels, dst_inputs, dst_labels,
                skip_existing, label,
            )
            futures[fut] = label

        with tqdm(total=total, desc="  files", unit="file") as pbar:
            for fut in as_completed(futures):
                status = fut.result()
                pbar.set_postfix_str(status[:60])
                pbar.update(1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base     = Path(__file__).parent          # korean_counseling_dataset/
    src_root = base
    dst_root = base.parent / "english_counseling_dataset"

    print(f"Source : {src_root}")
    print(f"Output : {dst_root}")
    dst_root.mkdir(parents=True, exist_ok=True)

    _error_log_path = dst_root / "error_log.json"

    for split in ("train", "valid"):
        src_split = src_root / split
        dst_split = dst_root / split
        if not src_split.exists():
            print(f"[SKIP] {src_split} not found.")
            continue
        print(f"\n=== Processing split: {split} ===")
        process_split(src_split, dst_split, skip_existing=True)

    print("\nDone.")
