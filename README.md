# counseling-data

Preprocessing scripts that **translate and clean Korean psychological-counseling dialogue datasets into English**, then build **train/valid CSVs** for model training.
Translation uses a local LLM via Ollama (`qwen3:14b`).

> Data files (`.txt`, `.json`, `.csv`) are not included in this repository — scripts only.

## Datasets

| Dataset | Content | Labels |
|---|---|---|
| `counseling_dataset` | General counseling dialogues (`resource_X.txt` ↔ `label_X.json`) | Depression / anxiety / addiction scores, age, gender, summary, etc. |
| `adolescent_consultant_dataset` | Adolescent counseling dialogues (`inputs/{category}/0008.txt` ↔ `labels/{category}/0008.json`) | Crisis level, total score, suspected abuse, clinician's assessment, etc. |

The adolescent dataset has 5 categories (directory names are in Korean):
`정상군` (normal) / `관찰필요` (needs observation) / `상담필요` (needs counseling) / `응급` (emergency) / `학대의심` (suspected abuse).

## Pipeline

```
Korean source ──translate_to_english.py──▶ English translation ──(cleanup scripts)──▶ build_*_csv.py ──▶ train.csv / valid.csv
```

## Main Code

### Translation — `translate_to_english.py` (both datasets)
- Translates Korean `.txt` dialogues to English, along with the text fields of the paired `.json` label. Numeric, ID, and categorical fields are kept as-is.
- Long dialogues are split into chunks with an `[N]` index on every line. Output is realigned by index, and the last 20 translated lines of the previous chunk are passed as context so the translation flows continuously.
- If the line count doesn't match, it retries; if it keeps failing, the chunk is split in half. Lines that still contain Korean (or other non-English) characters are re-translated.
- Files are processed in parallel, already-translated files are skipped, and failures are logged to `error_log.json`.

### `counseling_dataset/`
| File | Purpose |
|---|---|
| `korean_counseling_dataset/translate_to_english.py` | Translates dialogues plus `paragraph_text` and `summary` into English |
| `korean_counseling_dataset/scan_sizes.py` | Reports dialogue length stats (characters, estimated tokens) — used to size the LLM context window |
| `korean_counseling_dataset/build_korean_csv.py` | Builds `korean_train.csv` / `korean_valid.csv` (one row per session) |
| `english_counseling_dataset/find_non_english.py` | Finds leftover non-English characters in translations and logs file + line numbers to `error_log.json` |
| `english_counseling_dataset/build_english_csv.py` | Builds `english_train.csv` / `english_valid.csv` from the translations |

### `adolescent_consultant_dataset/`
| File | Purpose |
|---|---|
| `korean_.../translate_to_english.py` | Translates dialogues plus the clinician's assessment and comments into English |
| `korean_.../correct_english_dataset.py` | Uses the LLM to fix first-person pronouns (I/me/my → you/your) in counselor lines → writes `..._corrected/` |
| `english_.../sample_check.py` | Inspects TXT vs. JSON mismatches in the corrected set (swapped speakers, pronouns, shifted lines) |
| `english_.../fix_english_dataset.py` | Rule-based (no LLM) fix for 5 kinds of TXT/JSON mismatches → writes `post-train/`, `post-valid/` |
| `fix_audio_chunks.py` | Restores `audio` chunks lost during de-duplication, using timestamps from the Korean source |
| `*/build_adolescent_csv.py` | Builds train/valid CSVs — one row per session (JSON `info` fields + category + full dialogue) |

## Usage

```bash
pip install ollama tqdm pandas
ollama pull qwen3:14b
OLLAMA_NUM_PARALLEL=4 ollama serve   # for parallel translation

python counseling_dataset/korean_counseling_dataset/translate_to_english.py
python counseling_dataset/english_counseling_dataset/build_english_csv.py
```

Each script locates its `train/` and `valid/` folders relative to its own location.
Exception: `fix_audio_chunks.py` and `sample_check.py` have hard-coded Windows absolute paths — edit `BASE` / `base` at the top of each script before running.
