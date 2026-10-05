import json, re, hashlib, random
from pathlib import Path
from collections import Counter, defaultdict

# ---------- Config ----------
RAW_DIR = Path("data/raw/engineering")
OUT_DIR = Path("data/processed/engineering")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "policies":  RAW_DIR / "policies.pdf",
    "curricula": RAW_DIR / "curricula.pdf",
    "catalog":   RAW_DIR / "catalog.pdf",
}

CHUNK_CHARS = 800  # Smaller chunks = more precise retrieval
OVERLAP_CHARS = 200  # More overlap = less info lost at boundaries
LOW_TEXT_THRESHOLD = 60  # chars; below this likely scanned/table-only page

COURSE_CODE_LINE_RE = re.compile(r"^\s*([A-Z]{2,6}\s*[-]?\s*\d{3,4}[A-Z]?)\b")
COURSE_CODE_ANY_RE  = re.compile(r"\b[A-Z]{2,6}\s*[-]?\s*\d{3,4}[A-Z]?\b")

# ---------- Extraction backend ----------
def extract_pages(pdf_path: Path):
    # Prefer PyMuPDF if installed (often faster/better), fallback to pdfplumber
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(doc.page_count):
            pages.append(doc.load_page(i).get_text("text") or "")
        return pages, "pymupdf"
    except Exception:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                pages.append(page.extract_text(x_tolerance=1.5) or "")
        return pages, "pdfplumber"

# ---------- Cleaning / chunking ----------
def normalize(text: str) -> str:
    text = text.replace("\u00ad", "")  # soft hyphen
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def fixed_chunks(text: str, max_chars=CHUNK_CHARS, overlap=OVERLAP_CHARS):
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunk = text[i:j].strip()
        if chunk:
            out.append(chunk)
        if j == len(text):
            break
        i = max(0, j - overlap)
    return out

def split_catalog_blocks(page_text: str):
    """
    Use general fixed-size chunking for catalog.
    This keeps the solution general and works for any document type.
    """
    return fixed_chunks(page_text)

def strip_repeated_headers_footers(pages):
    """
    Remove very frequent first/last non-empty lines (common headers/footers).
    Conservative: only remove lines that occur on >=30% pages.
    """
    first = []
    last = []
    for t in pages:
        lines = [ln.strip() for ln in (t or "").splitlines() if ln.strip()]
        if not lines:
            continue
        first.append(lines[0])
        last.append(lines[-1])

    total = max(1, len(pages))
    c_first = Counter(first)
    c_last  = Counter(last)

    cut = int(0.30 * total)
    bad_first = {k for k, v in c_first.items() if v >= cut and len(k) <= 80}
    bad_last  = {k for k, v in c_last.items()  if v >= cut and len(k) <= 80}

    cleaned = []
    for t in pages:
        lines = [ln.rstrip() for ln in (t or "").splitlines()]
        # remove first occurrence if it matches
        i = 0
        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].strip() in bad_first:
            lines[i] = ""
        # remove last occurrence if it matches
        j = len(lines) - 1
        while j >= 0 and not lines[j].strip():
            j -= 1
        if j >= 0 and lines[j].strip() in bad_last:
            lines[j] = ""
        cleaned.append("\n".join(lines))
    return cleaned

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:12]

# ---------- Main build ----------
def build():
    for k, p in FILES.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing {k}: {p}")

    all_chunks = []
    report = {
        "school": "engineering",
        "backend_used": {},
        "stats": {},
        "warnings": [],
    }

    extracted_course_codes = Counter()

    for doc_type, pdf_path in FILES.items():
        pages, backend = extract_pages(pdf_path)
        report["backend_used"][doc_type] = backend

        # strip frequent headers/footers to improve splitting
        pages = strip_repeated_headers_footers(pages)

        doc_stats = {
            "file": str(pdf_path),
            "total_pages": len(pages),
            "pages_with_text": 0,
            "low_text_pages": 0,
            "total_chars": 0,
            "chunks": 0,
            "low_text_page_examples": [],
        }

        prev_catalog_chunk = None  # for stitching catalog continuations across pages

        for pno, raw in enumerate(pages, start=1):
            text = normalize(raw or "")
            doc_stats["total_chars"] += len(text)

            if len(text) < LOW_TEXT_THRESHOLD:
                doc_stats["low_text_pages"] += 1
                if len(doc_stats["low_text_page_examples"]) < 8:
                    doc_stats["low_text_page_examples"].append(pno)
                continue

            doc_stats["pages_with_text"] += 1

            # Gather course codes for quick sanity checks
            for code in COURSE_CODE_ANY_RE.findall(text.upper()):
                extracted_course_codes[re.sub(r"\s+", "", code)] += 1

            if doc_type == "catalog":
                blocks = split_catalog_blocks(text)
                if not blocks:
                    blocks = fixed_chunks(text)

                # Stitch: if page starts with non-course content and we had a previous chunk,
                # append the first block to previous chunk (common for cross-page course entries).
                if prev_catalog_chunk and blocks:
                    first_line = blocks[0].splitlines()[0].strip() if blocks[0].splitlines() else ""
                    if not COURSE_CODE_LINE_RE.match(first_line):
                        # extend previous chunk
                        prev_catalog_chunk["text"] += "\n" + blocks[0]
                        prev_catalog_chunk["page_end"] = pno
                        prev_catalog_chunk["id"] = f"{prev_catalog_chunk['id']}_cont"
                        blocks = blocks[1:]

                for bi, b in enumerate(blocks):
                    chunk = {
                        "id": f"eng_{doc_type}_{pno:04d}_{bi:02d}_{sha1(b)}",
                        "school": "engineering",
                        "doc_type": doc_type,
                        "source_file": str(pdf_path),
                        "page_start": pno,
                        "page_end": pno,
                        "text": b,
                    }
                    all_chunks.append(chunk)
                    prev_catalog_chunk = chunk

            else:
                chunks = fixed_chunks(text)
                for ci, ck in enumerate(chunks):
                    chunk = {
                        "id": f"eng_{doc_type}_{pno:04d}_{ci:02d}_{sha1(ck)}",
                        "school": "engineering",
                        "doc_type": doc_type,
                        "source_file": str(pdf_path),
                        "page_start": pno,
                        "page_end": pno,
                        "text": ck,
                    }
                    all_chunks.append(chunk)

        doc_stats["chunks"] = sum(1 for c in all_chunks if c["doc_type"] == doc_type)
        report["stats"][doc_type] = doc_stats

        # Warnings
        if doc_stats["total_pages"] > 0:
            low_ratio = doc_stats["low_text_pages"] / doc_stats["total_pages"]
            if low_ratio >= 0.15:
                report["warnings"].append(
                    f"{doc_type}: {doc_stats['low_text_pages']}/{doc_stats['total_pages']} pages have little/no extractable text "
                    f"(likely scanned or table-only). Consider OCR/table handling later."
                )

    # Write outputs
    chunks_path = OUT_DIR / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    # Preview for human validation
    preview_path = OUT_DIR / "preview.md"
    top_codes = extracted_course_codes.most_common(30)
    by_type = defaultdict(list)
    for c in all_chunks:
        by_type[c["doc_type"]].append(c)

    rnd = random.Random(42)
    lines = []
    lines.append("# Engineering dataset preview\n")
    lines.append("## Top extracted course-code-like tokens\n")
    lines.append("\n".join([f"- {code}: {cnt}" for code, cnt in top_codes]) or "- (none found)")
    lines.append("\n\n## Sample chunks\n")

    for dt in ["policies", "curricula", "catalog"]:
        lines.append(f"\n### {dt.upper()}\n")
        sample = by_type.get(dt, [])
        if not sample:
            lines.append("_No chunks produced._\n")
            continue
        picks = [sample[0], sample[len(sample)//2], sample[-1]]
        picks += [sample[i] for i in rnd.sample(range(len(sample)), k=min(2, len(sample)))]
        seen = set()
        for c in picks:
            if c["id"] in seen:
                continue
            seen.add(c["id"])
            lines.append(f"**[{c['doc_type']} p{c['page_start']}-{c['page_end']}]** `{c['id']}`\n\n")
            excerpt = c["text"]
            excerpt = excerpt[:1200] + ("..." if len(excerpt) > 1200 else "")
            lines.append(excerpt + "\n\n---\n")

    preview_path.write_text("\n".join(lines), encoding="utf-8")

    print("Done.")
    print(f"- chunks:  {chunks_path}")
    print(f"- report:  {report_path}")
    print(f"- preview: {preview_path}")
    if report["warnings"]:
        print("\nWarnings:")
        for w in report["warnings"]:
            print(" -", w)

if __name__ == "__main__":
    build()
