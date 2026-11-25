"""Map bundle transcripts to per-report FT decision files (labels from index.json)."""
import json, sys, os, re, glob
sys.path.insert(0, os.path.dirname(__file__))
from extract_decisions import find_json
tasks, work = sys.argv[1], sys.argv[2]
idx = {os.path.basename(e["prompt_file"])[:-4]: e["label"] for e in json.load(open(f"{work}/prompts_ft/index.json"))}
idx.update({os.path.basename(e["prompt_file"])[:-4]: e["label"] for f in sys.argv[3:] for e in json.load(open(f))})
meta = {os.path.basename(m["bundle"])[:-4]: m for m in json.load(open(f"{work}/ft_bundles/bundles.json"))}
os.makedirs(f"{work}/ft_decisions", exist_ok=True)
done, missing = set(), []
for p in glob.glob(f"{tasks}/*.output"):
    txt = open(p).read()
    m = re.search(r"Read the file \S*/ft_bundles/(FT[AB]_bundle\d+)\.txt and follow", txt)
    m2 = re.search(r"Read the file \S*/(prompts_ft\w*)/(FT\w+_R\d+)\.txt and follow", txt)
    if not m and not m2: continue
    found = None
    for line in txt.splitlines():
        try: ev = json.loads(line)
        except Exception: continue
        if ev.get("type") != "assistant" and ev.get("message", {}).get("role") != "assistant": continue
        r = find_json(ev.get("message", ev))
        if r: found = r
    if m:
        b = meta[m.group(1)]; role = b["role"]; want = b["ids"]
    else:
        role, rid = m2.group(2).split("_"); want = [rid]
    got = {d.get("id"): d for d in (found or {}).get("decisions", [])}
    for rid in want:
        key = f"{role}_{rid}"
        if rid in got and key in idx:
            json.dump({"label": idx[key], "decisions": [got[rid]]}, open(f"{work}/ft_decisions/{idx[key].replace(':', '_')}.json", "w")); done.add(key)
        else: missing.append(key)
print("saved", len(done), "missing", len(missing), sorted(missing)[:20])
