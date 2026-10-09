import sys
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path

base = Path(__file__).parent
txt_files = list(base.rglob("*.txt"))

results = []
for p in txt_files:
    try:
        text = p.read_text(encoding="utf-8").strip()
        chars = len(text)
        tokens = int(chars / 1.5)
        results.append((chars, tokens, p))
    except Exception as e:
        print(f"ERROR: {p}: {e}")

results.sort(reverse=True)
print(f"Total .txt files: {len(results)}\n")
print(f"{'Rank':<5} {'Chars':>10} {'Est.Tokens':>12}  Filename")
print("-" * 80)
for i, (chars, tokens, p) in enumerate(results[:20], 1):
    print(f"{i:<5} {chars:>10,} {tokens:>12,}  {p.name}")

print("\n--- Summary ---")
print(f"MAX chars  : {results[0][0]:,}")
print(f"MAX tokens : {results[0][1]:,}")
print(f"Max file   : {results[0][2].relative_to(base)}")
print(f"MIN chars  : {results[-1][0]:,}")
print(f"MIN tokens : {results[-1][1]:,}")
print(f"MEAN chars : {sum(r[0] for r in results) // len(results):,}")
print(f"MEAN tokens: {sum(r[1] for r in results) // len(results):,}")
