"""Fetch open-access PDFs for advanced records: arXiv first, then OpenAlex best OA PDF. Logs every outcome."""
import json, re, os, time, urllib.request, urllib.parse, sys
W = "sr_work"; OUT = sys.argv[1]
recs = {r["id"]: r for r in json.load(open(f"{W}/records.json"))["unique"]}
raw = json.load(open(f"{W}/raw_records.json"))
dec = json.load(open(f"{W}/decisions.json"))
adv = [i for i, v in dec.items() if v["final"]["d"] in ("include", "unclear")]
UA = {"User-Agent": "Mozilla/5.0 (sr-fulltext-fetch)"}
def get(url, t=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=t).read()
log = {}
for rid in sorted(adv):
    dst = f"{OUT}/{rid}.pdf"
    if os.path.exists(dst) and open(dst, "rb").read(4) == b"%PDF": log[rid] = "cached"; continue
    r = recs[rid]; urls = []
    m = re.search(r"arxiv\.(\d{4}\.\d{4,5})", (r.get("doi") or "").lower())
    arx = m.group(1) if m else None
    for mi in eval(r["members"]) if isinstance(r["members"], str) else r["members"]:
        for u in raw[mi].get("UR", []):
            mm = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", u)
            if mm: arx = arx or mm.group(1)
    if arx: urls.append(f"https://arxiv.org/pdf/{arx}")
    if r.get("doi") and not arx:
        try:
            w = json.loads(get("https://api.openalex.org/works/doi:" + urllib.parse.quote(r["doi"])))
            for loc in [w.get("best_oa_location")] + (w.get("locations") or []):
                if loc and loc.get("pdf_url"): urls.append(loc["pdf_url"])
        except Exception as e: pass
        time.sleep(1)
    ok = False
    for u in urls:
        try:
            b = get(u)
            if b[:4] == b"%PDF": open(dst, "wb").write(b); ok = True; log[rid] = "ok:" + u; break
        except Exception as e: pass
        time.sleep(3)
    if not ok: log[rid] = "not_retrieved:" + ";".join(urls)
json.dump(log, open("fulltext_fetch_log.json", "w"), indent=1)
print(sum(v.startswith(("ok", "cached")) for v in log.values()), "retrieved of", len(adv))
