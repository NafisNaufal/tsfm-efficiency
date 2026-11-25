"""Extract wrapped text from retrieved PDFs (PyMuPDF), with page markers; writes map CSV for prepare_fulltext.py."""
import fitz, glob, os, sys, textwrap, csv
src, dst, mapf = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for p in sorted(glob.glob(src + "/*.pdf")):
    rid = os.path.basename(p)[:-4]
    try: doc = fitz.open(p)
    except Exception as e: print("fail", rid, e); continue
    out = [f"### FULL TEXT of record {rid} (text extracted from the PDF with PyMuPDF; figures omitted; tables as plain text)"]
    for i, page in enumerate(doc):
        out.append(f"=== PAGE {i+1} ===")
        for para in page.get_text("text").split("\n"):
            para = para.strip()
            if para: out.extend(textwrap.wrap(para, 150) or [""])
    out.append("=== END OF DOCUMENT ===")
    f = f"{dst}/{rid}.txt"; open(f, "w").write("\n".join(out)); rows.append((rid, f))
with open(mapf, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["id", "pdf"]); w.writerows(rows)
print(len(rows), "extracted")
