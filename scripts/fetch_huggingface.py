"""Download Arabic E-Book Corpus from HuggingFace API"""
import urllib.request
import json
import os
import sys

DATASET = "mohres/The_Arabic_E-Book_Corpus"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "downloads")
os.makedirs(OUT_DIR, exist_ok=True)

# Check total rows
url = f"https://datasets-server.huggingface.co/rows?dataset={DATASET.replace('/', '%2F')}&config=default&split=train&offset=0&length=1"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
resp = urllib.request.urlopen(req, timeout=30)
data = json.loads(resp.read())
total = data.get("num_rows_total", 0)
print(f"Total rows: {total}")
if data.get("rows"):
    keys = list(data["rows"][0].get("row", {}).keys())
    print(f"Columns: {keys}")

# Download all rows in batches
out_file = os.path.join(OUT_DIR, "hindawi_all.txt")
if os.path.exists(out_file) and os.path.getsize(out_file) > 1_000_000:
    print(f"Already exists: {os.path.getsize(out_file) // 1000}KB")
    sys.exit(0)

BATCH = 100
count = 0
with open(out_file, "w", encoding="utf-8") as f:
    for offset in range(0, total, BATCH):
        try:
            url = f"https://datasets-server.huggingface.co/rows?dataset={DATASET.replace('/', '%2F')}&config=default&split=train&offset={offset}&length={BATCH}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=60)
            data = json.loads(resp.read())
            for row in data.get("rows", []):
                r = row.get("row", {})
                # Try different text fields
                text = r.get("text", "") or r.get("content", "") or r.get("body", "")
                if not text:
                    # Some rows have nested structures
                    for k, v in r.items():
                        if isinstance(v, str) and len(v) > 100:
                            text = v
                            break
                if text and len(text) > 50:
                    f.write(text.strip() + "\n\n")
                    count += 1
            print(f"  offset={offset}, books={count}", flush=True)
        except Exception as e:
            print(f"  Error at {offset}: {e}")
        import time
        time.sleep(1)

print(f"Done: {count} books to {out_file}")
if count > 0:
    size = os.path.getsize(out_file)
    print(f"Size: {size // 1000}KB ({size // 1_000_000}MB)")
