import os, json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

base = r"C:\Users\cwc\Desktop\cwc\TL\foundation data\adolescent_consultant_dataset\english_adolescent_consultant_dataset_corrected\train"
categories = ["정상군", "관찰필요", "상담필요", "응급", "학대의심"]

def load_txt(path):
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

def load_json_audio(path):
    with open(path, encoding="utf-8") as f:
        label = json.load(f)
    entries = []
    for domain in label.get("list", []):
        for item in domain.get("list", []):
            for entry in item.get("audio", []):
                entries.append((entry.get("type","?"), entry.get("text","").strip()))
    return entries

type_diff_samples = []
text_diff_you_i = []
text_diff_shift_files = []
extra_in_txt = []

for cat in categories:
    input_dir = os.path.join(base, "inputs", cat)
    label_dir = os.path.join(base, "labels", cat)
    if not os.path.exists(input_dir):
        continue
    for fname in sorted(os.listdir(input_dir)):
        if not fname.endswith(".txt"):
            continue
        case_id = fname.replace(".txt", "")
        txt_path = os.path.join(input_dir, fname)
        json_path = os.path.join(label_dir, case_id + ".json")
        if not os.path.exists(json_path):
            continue
        txt = load_txt(txt_path)
        jau = load_json_audio(json_path)
        n = max(len(txt), len(jau))
        file_extra = []
        file_type_diff = []
        has_shift = False
        for i in range(n):
            if i >= len(jau):
                file_extra.append((i, txt[i]))
            elif i >= len(txt):
                pass
            else:
                t_type, t_text = txt[i]
                j_type, j_text = jau[i]
                if t_text != j_text:
                    if "you" in t_text and ("I" in j_text or "my" in j_text):
                        text_diff_you_i.append((cat, case_id, i, t_text, j_text))
                    else:
                        has_shift = True
                elif t_type != j_type:
                    file_type_diff.append((i, t_type, j_type, t_text))
        if file_extra:
            extra_in_txt.append((cat, case_id, file_extra, txt, jau))
        if file_type_diff:
            type_diff_samples.append((cat, case_id, file_type_diff, txt, jau))
        if has_shift and not text_diff_shift_files:
            text_diff_shift_files.append((cat, case_id, txt, jau))

# ── CASE 1: TYPE_DIFF ──────────────────────────────────────────────────────
print("=" * 70)
print("CASE 1: TYPE_DIFF -- speaker label swapped (text is identical)")
print("=" * 70)
shown = 0
for cat, case_id, diffs, txt, jau in type_diff_samples:
    if shown >= 3:
        break
    first_i = diffs[0][0]
    start = max(0, first_i - 1)
    end = min(len(txt), first_i + len(diffs) + 1)
    print(f"\nFile: {cat}/{case_id}  ({len(diffs)} swapped lines)")
    print(f"{'idx':<5} {'TXT speaker':<12} {'JSON speaker':<12} text")
    print("-" * 72)
    for i in range(start, end):
        t_type, t_text = txt[i]
        j_type, j_text = jau[i] if i < len(jau) else ("?", "?")
        flag = " <<<" if t_type != j_type else ""
        t_spk = "Counselor" if t_type == "Q" else "Client"
        j_spk = "Counselor" if j_type == "Q" else "Client"
        print(f"{i:<5} {t_spk:<12} {j_spk:<12} {t_text[:50]}{flag}")
    shown += 1

# ── CASE 2a: TEXT_DIFF you->I ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("CASE 2a: TEXT_DIFF -- 'you' in TXT vs 'I' in JSON (pronoun error)")
print("=" * 70)
for cat, case_id, i, t, j in text_diff_you_i[:5]:
    print(f"\nFile: {cat}/{case_id}  line {i}")
    print(f"  TXT  (Counselor Q): {t}")
    print(f"  JSON (audio.text) : {j}")

# ── CASE 2b: TEXT_DIFF shift ──────────────────────────────────────────────
print("\n" + "=" * 70)
print("CASE 2b: TEXT_DIFF -- content shift (JSON audio offset by 1)")
print("=" * 70)
if text_diff_shift_files:
    cat, case_id, txt, jau = text_diff_shift_files[0]
    # find first mismatch index
    first_mm = next(i for i in range(min(len(txt),len(jau))) if txt[i][1] != jau[i][1])
    start = max(0, first_mm - 1)
    end = min(len(txt), first_mm + 8)
    print(f"\nFile: {cat}/{case_id}  (txt={len(txt)} lines, json={len(jau)} entries)")
    print(f"{'idx':<5} {'TXT spk':<10} {'TXT text':<48} {'JSON text'}")
    print("-" * 115)
    for i in range(start, end):
        t_type, t_text = txt[i] if i < len(txt) else ("?","?")
        j_type, j_text = jau[i] if i < len(jau) else ("?","?")
        flag = " <<<" if t_text != j_text else ""
        spk = "Counselor" if t_type == "Q" else "Client"
        print(f"{i:<5} {spk:<10} {t_text:<48} {j_text[:48]}{flag}")

# ── CASE 3: EXTRA_IN_TXT ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("CASE 3: EXTRA_IN_TXT -- lines in TXT with no audio entry in JSON")
print("=" * 70)
shown = 0
for cat, case_id, file_extra, txt, jau in extra_in_txt:
    if shown >= 3:
        break
    print(f"\nFile: {cat}/{case_id}  (txt={len(txt)}, json audio={len(jau)}, +{len(file_extra)} extra)")
    boundary = file_extra[0][0]
    start = max(0, boundary - 2)
    print(f"  {'idx':<5} {'speaker':<10} text")
    print("  " + "-" * 68)
    for i in range(start, len(txt)):
        t_type, t_text = txt[i]
        spk = "Counselor" if t_type == "Q" else "Client"
        tag = "  [NO JSON ENTRY]" if i >= boundary else ""
        print(f"  {i:<5} {spk:<10} {t_text[:58]}{tag}")
    shown += 1
