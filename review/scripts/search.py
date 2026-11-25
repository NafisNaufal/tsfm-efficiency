"""Documented, reproducible database search for the SR (run 2026-11-23).
Sources: OpenAlex (title+abstract search), Semantic Scholar bulk search, arXiv API.
Raw responses saved under raw/<source>/; combined RIS written to exports/."""
import urllib.parse, urllib.request, json, time, os, re, sys, xml.etree.ElementTree as ET
OUT = sys.argv[1]
os.makedirs(OUT + "/raw/openalex", exist_ok=True); os.makedirs(OUT + "/raw/s2", exist_ok=True); os.makedirs(OUT + "/raw/arxiv", exist_ok=True); os.makedirs(OUT + "/exports", exist_ok=True)
UA = {"User-Agent": "sr-search-script/1.0"}
def get(url, tries=10):
    for i in range(tries):
        time.sleep(2)
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode()
        except Exception as e:
            body = e.read()[:200] if hasattr(e, "read") else b""
            print("retry", i, str(e)[:80], body, file=sys.stderr)
            if getattr(e, "code", 0) in (400, 401, 403, 404): raise; time.sleep(10 * (i + 1))
    raise RuntimeError("failed " + url[:120])

# ---- query blocks (same concepts across databases) ----
OA_Q = {
 "Q1_tsfm_comparators": '("foundation model" OR "foundation models" OR "zero-shot" OR "in-context" OR "pretrained" OR "pre-trained") AND ("time series" OR forecasting) AND (oracle OR Bayes OR "context length" OR "context window" OR "random walk" OR martingale OR "white noise" OR "impulse response" OR Jacobian OR ARIMA OR GARCH OR Kalman OR "autoregressive process" OR "least squares" OR "synthetic")',
 "Q2_icl_dynamics": '("in-context") AND (transformer OR transformers OR attention) AND ("dynamical system" OR "dynamical systems" OR "time series" OR "autoregressive" OR Kalman OR "linear system" OR forecasting)',
}
S2_Q = {
 "Q1_tsfm_comparators": '("foundation model" | "foundation models" | "zero-shot" | "in-context" | pretrained | "pre-trained") + ("time series" | forecasting) + (oracle | Bayes | "context length" | "context window" | "random walk" | martingale | "white noise" | "impulse response" | Jacobian | ARIMA | GARCH | Kalman | "autoregressive process" | "least squares" | synthetic)',
 "Q2_icl_dynamics": '"in-context" + (transformer | transformers | attention) + ("dynamical system" | "dynamical systems" | "time series" | autoregressive | Kalman | "linear system" | forecasting)',
}
ARX_Q = {
 "Q1a": 'abs:"foundation model" AND abs:"time series"',
 "Q1b": 'abs:"foundation models" AND abs:"time series"',
 "Q1c": 'abs:"zero-shot" AND abs:"time series" AND abs:forecasting',
 "Q2a": 'abs:"in-context" AND abs:"dynamical systems"',
 "Q2b": 'abs:"in-context" AND abs:"time series"',
 "Q2c": 'abs:"in-context" AND abs:"dynamical system"',
}
recs = {}
def add(src, qn, r):
    key = (r.get("doi") or "").lower() or ("arxiv:" + r["arxiv"] if r.get("arxiv") else "") or r["title"].lower()
    r["sources"] = [f"{src}:{qn}"]
    if key in recs: recs[key]["sources"].append(f"{src}:{qn}"); return
    recs[key] = r

def inv_abstract(ii):
    if not ii: return ""
    pos = sorted((p, w) for w, ps in ii.items() for p in ps); return " ".join(w for _, w in pos)

counts = {}
for qn, q in OA_Q.items():
    n, page = 0, 0
    while True:
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"filter": f"title_and_abstract.search:{q},from_publication_date:2019-01-01,to_publication_date:2026-11-23", "per-page": 200, "page": page + 1})
        t = get(url); open(f"{OUT}/raw/openalex/{qn}_{page}.json", "w").write(t); d = json.loads(t); page += 1
        for w in d["results"]:
            ids = w.get("ids", {}); doi = (w.get("doi") or "").replace("https://doi.org/", "")
            arx = ""
            for loc in w.get("locations", []) or []:
                lu = (loc.get("landing_page_url") or "")
                m = re.search(r"arxiv\.org/abs/(\d{4}\.\d{4,5})", lu)
                if m: arx = m.group(1)
            m = re.match(r"10\.48550/arxiv\.(\d{4}\.\d{4,5})", doi.lower())
            if m: arx = m.group(1)
            add("openalex", qn, {"title": w.get("title") or "", "abstract": inv_abstract(w.get("abstract_inverted_index")), "year": w.get("publication_year"),
                "authors": [a["author"]["display_name"] for a in w.get("authorships", [])], "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name") or "",
                "doi": doi, "arxiv": arx, "type": w.get("type") or "", "url": w.get("id")})
            n += 1
        if len(d["results"]) < 200: break
    counts[f"openalex:{qn}"] = n
for qn, q in S2_Q.items():
    token, n, page = None, 0, 0
    while True:
        p = {"query": q, "year": "2019-2026", "fields": "title,abstract,year,authors,venue,externalIds,publicationTypes,url"}
        if token: p["token"] = token
        t = get("https://api.semanticscholar.org/graph/v1/paper/search/bulk?" + urllib.parse.urlencode(p)); open(f"{OUT}/raw/s2/{qn}_{page}.json", "w").write(t); d = json.loads(t); page += 1
        for w in d.get("data", []):
            ex = w.get("externalIds") or {}
            add("s2", qn, {"title": w.get("title") or "", "abstract": w.get("abstract") or "", "year": w.get("year"), "authors": [a.get("name") for a in w.get("authors") or []],
                "venue": w.get("venue") or "", "doi": ex.get("DOI") or "", "arxiv": ex.get("ArXiv") or "", "type": ",".join(w.get("publicationTypes") or []), "url": w.get("url")})
            n += 1
        token = d.get("token"); time.sleep(3)
        if not token: break
    counts[f"s2:{qn}"] = n
NS = {"a": "http://www.w3.org/2005/Atom"}
for qn, q in ARX_Q.items():
    start, n = 0, 0
    while True:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode({"search_query": q + " AND submittedDate:[201901010000 TO 202611232359]", "start": start, "max_results": 200})
        t = get(url); open(f"{OUT}/raw/arxiv/{qn}_{start}.xml", "w").write(t); root = ET.fromstring(t)
        es = root.findall("a:entry", NS)
        for e in es:
            aid = re.sub(r"v\d+$", "", e.find("a:id", NS).text.split("/abs/")[-1])
            add("arxiv", qn, {"title": " ".join(e.find("a:title", NS).text.split()), "abstract": " ".join(e.find("a:summary", NS).text.split()),
                "year": int(e.find("a:published", NS).text[:4]), "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)], "venue": "arXiv",
                "doi": "", "arxiv": aid, "type": "preprint", "url": "https://arxiv.org/abs/" + aid})
            n += 1
        start += 200; time.sleep(4)
        if len(es) < 200: break
    counts[f"arxiv:{qn}"] = n
json.dump({"queries": {"openalex": OA_Q, "s2": S2_Q, "arxiv": ARX_Q}, "counts": counts, "unique_prelim": len(recs), "date": "2026-11-23"}, open(f"{OUT}/search_log.json", "w"), indent=1)
# per-source RIS (dedup is done by prepare_records.py)
for src in ["openalex", "s2", "arxiv"]:
    with open(f"{OUT}/exports/{src}.ris", "w") as f:
        for r in recs.values():
            if not any(s.startswith(src + ":") for s in r["sources"]): continue
            f.write("TY  - " + ("JOUR" if src != "arxiv" else "GEN") + "\n")
            f.write("TI  - " + r["title"].replace("\n", " ") + "\n")
            for a in r["authors"][:30]: f.write("AU  - " + str(a) + "\n")
            if r["year"]: f.write(f"PY  - {r['year']}\n")
            if r["venue"]: f.write("T2  - " + r["venue"] + "\n")
            if r["doi"]: f.write("DO  - " + r["doi"] + "\n")
            if r["arxiv"]: f.write("UR  - https://arxiv.org/abs/" + r["arxiv"] + "\n")
            elif r.get("url"): f.write("UR  - " + str(r["url"]) + "\n")
            if r["type"]: f.write("M3  - " + r["type"] + "\n")
            if r["abstract"]: f.write("AB  - " + r["abstract"].replace("\n", " ") + "\n")
            f.write("ER  - \n\n")
print(json.dumps(counts, indent=1), "unique_prelim", len(recs))
