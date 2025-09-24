import magic
import multiprocessing as mp
import subprocess
import string
from ollama import Client
from rake_nltk import Rake
import docx
import re
import json, csv, os, time
import unicodedata
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None
import math
from collections import Counter, defaultdict


# change the model when switching pc/laptop
OLLAMA_HOST = "http://127.0.0.1:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"
EXTRA_DASHES = r"[–—−-]"  # en dash, em dash, minus, non-breaking hyphen
FORCE_RAKE = True


ollama_client = Client(host=OLLAMA_HOST)
# Fast preflight so we don't hang on generate() if the server isn't reachable
try:
    _ = ollama_client.list()  # pings /api/tags under the hood
except Exception as e:
    print(f"[ERR] Cannot reach Ollama at {OLLAMA_HOST}. Is the service running and listening on 127.0.0.1:11434?\n{e}")
    import sys
    sys.exit(1)

translator = str.maketrans('', '', string.punctuation)


def sent_tokenize(s: str) -> list[str]:
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+', s) if x.strip()]

def noun_phrases_or_rake(text: str) -> list[str]:
    # prefer spaCy noun chunks if available; fallback to RAKE
    if _NLP and not FORCE_RAKE:
        try:
            doc = _NLP(text)
            cands = [re.sub(EXTRA_DASHES, "-", nc.text.strip()) for nc in doc.noun_chunks]
        except Exception:
            cands = []
    else:
        r = Rake()
        r.extract_keywords_from_text(normalize_for_rake(text))
        cands = [re.sub(EXTRA_DASHES, "-", x.strip()) for x in r.get_ranked_phrases()]
    # normalize spacing
    cands = [re.sub(r"\s+", " ", c).strip(" :;,.\"'()[]{}") for c in cands if c]
    # dedupe (case-insensitive)
    seen = set(); uniq = []
    for c in cands:
        k = c.lower()
        if k not in seen:
            seen.add(k); uniq.append(c)
    return uniq

def build_corpus_stats(chunks: list[str]):
    """
    Build DF/TF stats over *phrases* (not single words) extracted per chunk.
    Returns (df, per_chunk_tf, chunk_norms, chunk_sents, position_maps)
    """
    df = Counter()
    per_chunk_tf = []
    chunk_norms = []
    chunk_sents = []
    position_maps = []  # earliest sentence index each phrase appears in

    for ch in chunks:
        sents = sent_tokenize(ch)
        chunk_sents.append(sents)
        ch_norm = normalize_for_match(ch)
        chunk_norms.append(ch_norm)

        cands = noun_phrases_or_rake(ch)
        tf = Counter()
        first_pos = {}
        for i, s in enumerate(sents):
            sn = normalize_for_match(s)
            for cand in cands:
                n = normalize_for_match(cand)
                if not n: 
                    continue
                # candidate appears in this sentence?
                if n in sn:
                    tf[cand] += 1
                    if cand not in first_pos:
                        first_pos[cand] = i
        per_chunk_tf.append(tf)
        for cand in tf.keys():
            df[cand] += 1
        position_maps.append(first_pos)

    return df, per_chunk_tf, chunk_norms, chunk_sents, position_maps

def tfidf_score(cand: str, tf: Counter, df: Counter, N: int) -> float:
    t = tf.get(cand, 0)
    if t == 0: 
        return 0.0
    d = df.get(cand, 0)
    idf = math.log(1 + (N / max(1, d)))
    return t * idf

def mmr_select(cands: list[str], cand_vecs: dict, k: int, diversity: float = 0.6):
    """
    Simple MMR using Jaccard over token sets (no embeddings needed).
    cand_vecs[c] = set(tokens)
    """
    if not cands:
        return []
    selected = []
    remaining = cands[:]
    # normalize similarity
    def sim(a, b):
        A, B = cand_vecs[a], cand_vecs[b]
        if not A or not B: return 0.0
        return len(A & B) / float(len(A | B))
    while remaining and len(selected) < k:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best, best_score = None, -1e9
        for c in remaining:
            rel = 1.0  # already baked into ranking earlier
            div = max(sim(c, s) for s in selected) if selected else 0.0
            score = diversity * rel - (1 - diversity) * div
            if score > best_score:
                best, best_score = c, score
        remaining.remove(best)
        selected.append(best)
    return selected


def _safe_generate(**kwargs):
    try:
        return ollama_client.generate(**kwargs)
    except Exception as e:
        # surface and skip this item instead of hanging the whole run
        raise

def _strip_code_fences(t: str) -> str:
    # remove ```...``` wrappers if the model adds them
    t = t.strip()
    if t.startswith("```"):
        t = re.sub(r"^```.*?\n", "", t, flags=re.S)  # drop opening fence line
        t = re.sub(r"\n```$", "", t, flags=re.S)     # drop closing fence
    return t

def _canonicalize_tags(t: str) -> str:
    """
    Normalize all plausible tag variants to canonical 'Q:', 'A:', 'E:' at line start.
    Handles: Q1:, Question:, A1:, Answer:, Evidence:, e:, etc.  Case-insensitive.
    """
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # standardize common labels to canonical
    reps = [
        (r'(?mi)^\s*question\s*[:\-–]\s*', 'Q: '),
        (r'(?mi)^\s*q\d*\s*[:\-–]\s*',      'Q: '),
        (r'(?mi)^\s*answer\s*[:\-–]\s*',    'A: '),
        (r'(?mi)^\s*a\d*\s*[:\-–]\s*',      'A: '),
        (r'(?mi)^\s*evidence\s*[:\-–]\s*',  'E: '),
        (r'(?mi)^\s*e\d*\s*[:\-–]\s*',      'E: '),
    ]
    for pat, repl in reps:
        t = re.sub(pat, repl, t)
    return t.strip()

def parse_qae(text: str) -> dict:
    """
    Parse ONE Q/A/E triple from model output.
    Returns: {'ok': bool, 'q': str|None, 'a': str|None, 'e': str|None, 'raw': str, 'missing': set()}
    - 'a' can be multi-line; we stop at the next 'E:' line
    - 'e' is unquoted (quotes removed if present)
    """
    raw = _canonicalize_tags(_strip_code_fences(text))

    # Grab Q
    m_q = re.search(r'(?m)^Q:\s*(.*)$', raw)
    q = m_q.group(1).strip() if m_q else None

    # Grab A (multi-line, non-greedy, until next E:)
    m_a = re.search(r'(?ms)^A:\s*(.*?)\n\s*E:\s*', raw)
    a = m_a.group(1).strip() if m_a else None

    # Grab E (to end of string)
    m_e = re.search(r'(?m)^E:\s*(.*)$', raw)
    e = m_e.group(1).strip() if m_e else None
    if e:
        # remove surrounding quotes if present
        e = e.strip().strip('"“”„').strip()

    missing = set()
    if not q: missing.add('Q')
    if not a: missing.add('A')
    if not e: missing.add('E')

    return {'ok': len(missing) == 0, 'q': q, 'a': a, 'e': e, 'raw': raw, 'missing': missing}

def parse_qae_items(text: str) -> list[dict]:
    """
    Parse MULTIPLE Q/A/E items if the model returns more than one.
    Splits on 'Q:' lines and applies parse_qae to each block.
    Returns a list of dicts like parse_qae().
    """
    raw = _canonicalize_tags(_strip_code_fences(text))
    # split while keeping 'Q:' with the following line by using a lookahead
    blocks = re.split(r'(?m)^(?=Q:\s*)', raw)
    items = []
    for b in blocks:
        if b.strip():
            items.append(parse_qae(b))
    return items

def normalize_for_rake(s: str) -> str:
    s = re.sub(EXTRA_DASHES, "-", s)
    s = s.replace("\u00a0", " ")  # nbsp -> space
    s = s.replace("-", " ")
    # drop section markers like (1) (2) (3)
    s = re.sub(r"\(\d+\)\s*", "", s)
    # drop header lines like "Reading Material 1 – ..."
    s = re.sub(r"^reading material\s*\d*\s*[-–—]?\s*.*$", "", s, flags=re.I | re.M)
    # lowercase + collapse spaces
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def normalize_for_match(s: str) -> str:
    s = re.sub(EXTRA_DASHES, " ", s)
    s = s.replace("\u00a0", " ")
    s = s.replace("-", " ")
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)  # strip punctuation (ASCII + Unicode)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_from_txt(file_path):
    with open(file_path) as doc:
        text = doc.read()
    return text


def extract_from_pdf_core(
    file_path,
    remove_headers=True,
    remove_footers=True,
    include_footnotes=True,
    detect_images=False,
    ocr_images=False,
    image_dir="extracted_images",
):
    """
    Robust text extraction with header/footer removal, reading-order sort,
    hyphen fix, and optional image OCR.

    Now hardened: any page-level extraction failure falls back to simpler
    text extraction so a single bad page can't crash the whole run.
    """
    import os
    import fitz  # PyMuPDF
    from statistics import median

    TOP_FRACTION = 0.08
    BOTTOM_FRACTION = 0.08
    FOOTNOTE_ZONE = 0.25
    REPEAT_THRESHOLD = 0.60
    Y_TOL = 4.0
    MIN_HEADER_LEN = 8
    MIN_FOOTER_LEN = 3

    # -------- open safely --------
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        # as a last resort, tell caller we couldn't parse
        raise RuntimeError(f"[PDF OPEN ERROR] {file_path}: {e}")

    try:
        pages_data = []
        top_candidates = []
        bot_candidates = []

        # First pass: gather candidate headers/footers
        for pno in range(len(doc)):
            page = doc[pno]
            width, height = page.rect.width, page.rect.height
            top_y = height * TOP_FRACTION
            bot_y = height * (1.0 - BOTTOM_FRACTION)

            # --- robust: if dict extraction crashes, skip to simple text ---
            try:
                d = page.get_text("dict")
            except Exception:
                # minimal fallback for header/footer detection
                d = {"blocks": []}

            lines = []
            for block in d.get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    buff, sizes = [], []
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if t:
                            buff.append(t)
                            try:
                                sizes.append(float(span.get("size", 0)))
                            except Exception:
                                pass
                    txt = "".join(buff).strip()
                    if not txt:
                        continue
                    bbox = line.get("bbox", [0, 0, 0, 0])
                    avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
                    lines.append((txt, bbox, avg_size))

            top_local, bot_local = [], []
            for txt, (x0, y0, x1, y1), _sz in lines:
                if y0 <= top_y and len(txt) >= MIN_HEADER_LEN:
                    norm = re.sub(r"\s+", " ", txt).strip()
                    top_local.append(norm)
                elif y1 >= bot_y and len(txt) >= MIN_FOOTER_LEN:
                    norm = re.sub(r"\s+", " ", txt).strip()
                    bot_local.append(norm)

            top_candidates.append(set(top_local))
            bot_candidates.append(set(bot_local))

            pages_data.append({
                "size": (width, height),
                "page": page,
            })

        def decide_repeating(candidates_list, threshold, page_count):
            from collections import Counter
            c = Counter()
            for s in candidates_list:
                c.update(s)
            min_pages = int(page_count * threshold + 0.5)
            return set([s for s, n in c.items() if n >= min_pages])

        headers = decide_repeating(top_candidates, REPEAT_THRESHOLD, len(doc)) if remove_headers else set()
        footers = decide_repeating(bot_candidates, REPEAT_THRESHOLD, len(doc)) if remove_footers else set()

        body_parts, footnotes_all = [], []

        if detect_images:
            os.makedirs(image_dir, exist_ok=True)

        # Second pass: extract body/footnotes with per-page fallback
        for pno, pdata in enumerate(pages_data):
            page = pdata["page"]
            width, height = pdata["size"]

            # Optional image export (wrapped)
            if detect_images:
                try:
                    for img in page.get_images(full=True):
                        xref = img[0]
                        try:
                            pix = page.get_pixmap(xref=xref)
                            out = os.path.join(image_dir, f"p{pno+1}_x{xref}.png")
                            pix.save(out)
                        except Exception:
                            try:
                                dimg = doc.extract_image(xref)
                                ext = dimg.get("ext", "png")
                                img_bytes = dimg.get("image")
                                with open(os.path.join(image_dir, f"p{pno+1}_x{xref}.{ext}"), "wb") as f:
                                    f.write(img_bytes)
                            except Exception:
                                pass
                except Exception:
                    pass

            # Try rich layout first
            lines = []
            try:
                d = page.get_text("dict")
                for block in d.get("blocks", []):
                    if block.get("type", 0) != 0:
                        continue
                    for line in block.get("lines", []):
                        buff, sizes = [], []
                        for span in line.get("spans", []):
                            t = span.get("text", "")
                            if t:
                                buff.append(t)
                                try:
                                    sizes.append(float(span.get("size", 0)))
                                except Exception:
                                    pass
                        txt = "".join(buff).strip()
                        if not txt:
                            continue
                        bbox = line.get("bbox", [0, 0, 0, 0])
                        avg_size = (sum(sizes) / len(sizes)) if sizes else 0.0
                        lines.append((txt, bbox, avg_size))
            except Exception:
                # Fallback: plain text only
                try:
                    plain = page.get_text("text") or ""
                    if plain.strip():
                        body_parts.append(plain.strip())
                    continue
                except Exception:
                    # Worst-case: skip this page
                    continue

            # sort lines (layout-aware)
            def sort_key(item):
                _txt, (x0, y0, x1, y1), _sz = item
                return (round(y0 / Y_TOL), x0)
            lines.sort(key=sort_key)

            page_body, page_foot = [], []
            for txt, (x0, y0, x1, y1), sz in lines:
                norm = re.sub(r"\s+", " ", txt).strip()
                if remove_headers and norm in headers:
                    continue
                if remove_footers and norm in footers:
                    continue
                # drop bare page numbers close to margins
                if (y0 <= height * TOP_FRACTION or y1 >= height * (1.0 - BOTTOM_FRACTION)) and re.fullmatch(r"\d{1,4}", norm):
                    continue

                if include_footnotes and (y1 >= height * (1.0 - FOOTNOTE_ZONE)):
                    page_foot.append((txt, sz))
                else:
                    page_body.append(txt)

            page_text = "\n".join(page_body)
            page_text = re.sub(r"(\w)-\n(\w)", r"\1\2", page_text)
            page_text = re.sub(r"[ \t]+\n", "\n", page_text)
            page_text = re.sub(r"\n{3,}", "\n\n", page_text)

            if page_text.strip():
                body_parts.append(page_text.strip())

            if include_footnotes and page_foot:
                sizes = [sz for _t, sz in page_foot if sz > 0]
                cutoff = (median(sizes) * 0.8) if sizes else 8.0
                notes = [t for t, sz in page_foot if sz > 0 and sz <= cutoff]
                if notes:
                    footnotes_all.extend([f"[p.{pno+1}] {n}" for n in notes])

        # Optional OCR (kept but wrapped)
        if detect_images and ocr_images:
            try:
                import pytesseract
                from PIL import Image
                ocr_texts = []
                for fn in sorted(os.listdir(image_dir)):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp")):
                        ocr_texts.append(pytesseract.image_to_string(Image.open(os.path.join(image_dir, fn))))
                if ocr_texts:
                    body_parts.append("\n\n--- OCR (images) ---\n" + "\n\n".join(t.strip() for t in ocr_texts if t.strip()))
            except Exception:
                pass

        final_text = "\n\n".join(bp for bp in body_parts if bp)
        if include_footnotes and footnotes_all:
            final_text += "\n\n--- FOOTNOTES ---\n" + "\n".join(footnotes_all)
        return final_text

    finally:
        try:
            doc.close()
        except Exception:
            pass


def extract_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for p in doc.paragraphs:
        text += '\n' + p.text

    for table in doc.tables:
        text += '\n'
        for row in table.rows:
            text += '\n'
            for cell in row.cells:
                text += cell.text
                text += ","
    return text


CTRL_RE = re.compile(r'[\x00-\x1F\x7F]')

def clean_text(s: str) -> str:
    if not s: return ""
    s = unicodedata.normalize('NFKC', s)
    # common ligatures / punctuation / artifacts
    s = (s.replace('ﬂ','fl')
           .replace('ﬁ','fi')
           .replace('ﬀ','ff')
           .replace('ﬃ','ffi')
           .replace('ﬄ','ffl')
           .replace('—','-').replace('–','-').replace('−','-')
           .replace('“','"').replace('”','"').replace("’","'")
           .replace('×','x'))  # or '×' -> 'x' for consistency
    s = CTRL_RE.sub('', s)
    return s


def _fitz_child_worker(path, kwargs, q):
    """
    Runs MuPDF extraction in a separate process so crashes don't kill the parent.
    Put result in a queue; on failure, put ''.
    """
    try:
        text = extract_from_pdf_core(path, **kwargs)
        q.put(text or "")
    except Exception:
        q.put("")

def _fallback_pdfminer(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        return pdfminer_extract_text(path) or ""
    except Exception:
        return ""

def _fallback_pypdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n\n".join((p.extract_text() or "") for p in reader.pages) or ""
    except Exception:
        return ""

def _fallback_pdftotext_cli(path: str) -> str:
    # optional: use Poppler CLI if installed
    try:
        out = subprocess.check_output(["pdftotext", "-layout", path, "-"], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace")
    except Exception:
        return ""

def extract_from_pdf_safe(path: str, timeout_sec: int = 45, **kwargs) -> str:
    """
    Safe front door for PDF extraction.
    1) Spawn a child process to run MuPDF.
    2) If child crashes/hangs, terminate and fall back to pdfminer/pypdf/CLI.
    """
    q = mp.Queue()
    p = mp.Process(target=_fitz_child_worker, args=(path, kwargs, q))
    p.start()
    p.join(timeout_sec)

    text = ""
    if p.is_alive():
        # hung -> terminate
        try: p.terminate()
        except Exception: pass
        p.join(2)

    if p.exitcode == 0:
        try:
            text = q.get_nowait()
        except Exception:
            text = ""

    if text.strip():
        print("[INFO] PDF text via: MuPDF (child)")
        return text

    # --- fallbacks ---
    text = _fallback_pdfminer(path)
    if text.strip():
        print("[INFO] PDF text via: pdfminer.six")
        return text

    text = _fallback_pypdf(path)
    if text.strip():
        print("[INFO] PDF text via: pypdf")
        return text

    text = _fallback_pdftotext_cli(path)
    print("[INFO] PDF text via: pdftotext CLI")
    return text


def input_content():
    while True:
        mode = input("[T]ext or [F]ile: ").lower()
        if (mode == "text" or mode == "t"):
            mode = 1
            break
        elif (mode == "file" or mode == "f"):
            mode = 2
            break
    return mode


def extract_content(mode):
    if mode == 1:
        return input("Please paste the content here: ")

    # mode == 2
    path = input("Please input the filepath here: ").strip()
    if not os.path.exists(path):
        print(f"[ERR] File not found: {path}")
        return extract_content(mode)

    # Show the absolute path so you know exactly what you're opening
    abs_path = os.path.abspath(path)
    print(f"[INFO] Opening: {abs_path}")

    try:
        ctype = magic.from_file(abs_path, mime=True)
        print(f"[INFO] Detected MIME type: {ctype}")
    except Exception as e:
        print(f"[WARN] libmagic failed on {abs_path}: {e}")
        # Fallback by extension
        ext = os.path.splitext(abs_path)[1].lower()
        if ext in {".txt"}:
            ctype = "text/plain"
        elif ext in {".pdf"}:
            ctype = "application/pdf"
        elif ext in {".docx"}:
            ctype = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            print(f"[ERR] Unsupported type by extension: {ext} (accepted: pdf, docx, txt)")
            return extract_content(mode)

    if ctype == "text/plain":
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ctype == "application/pdf":
        return extract_from_pdf_safe(abs_path)

    if ctype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_from_docx(abs_path)

    print(f"[ERR] Unsupported type: {ctype} (accepted: pdf, docx, txt)")
    return extract_content(mode)




def split_sentences(text):
    # split on sentence end *or* line breaks
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def chunks(chunk_size, sentences):
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= chunk_size:
            current_chunk += " " + sentence.strip()
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence.strip()
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def join_chunks(chunk_size, text, overlap_sents=2):
    """
    Sentence-based windows with overlap so we don't miss boundary info.
    overlap_sents = how many sentences to carry over to the next chunk.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks = []
    cur = []
    cur_len = 0

    i = 0
    while i < len(sentences):
        s = sentences[i].strip()
        if not s:
            i += 1
            continue

        if cur_len + len(s) + 1 <= chunk_size:
            cur.append(s)
            cur_len += len(s) + 1
            i += 1
        else:
            if cur:
                chunks.append(" ".join(cur).strip())
            # start next window with overlap_sents tail from previous
            tail = cur[-overlap_sents:] if overlap_sents > 0 else []
            cur = tail[:]  # copy
            cur_len = sum(len(x) + 1 for x in cur)

    if cur:
        chunks.append(" ".join(cur).strip())

    return chunks


QUESTION_TEMPLATES = [
    "How does {A} influence or determine {X} in the context described?",
    "Why does {A} matter for {X}, and what are the key consequences discussed?",
    "Compare {A} with an alternative approach/structure mentioned; what trade-offs emerge?",
    "Under what conditions does {A} guarantee or fail to guarantee {X}?",
    "Explain the mechanism by which {A} leads to {X}, citing a concrete detail."
]

def build_qg_prompt_variants(chunk_text: str, anchor: str) -> str:
    # ask for multiple candidates in one call to reduce latency
    # Model returns three Q/A/E blocks, each following the same format.
    template_hint = (
        "Pick three *different* types from: causal, comparative, conditional, mechanism.\n"
        "Vary the opener (How/Why/Explain/Compare/Under what conditions).\n"
    )
    return (
        "You will produce THREE candidate study items from the chunk ONLY.\n"
        "Each item must follow exactly:\n"
        "Q: <one open question>\n"
        "A: <3–4 sentences: cover a core idea; add one concrete detail from the chunk NOT used as E; add a consequence/implication; mention any stakeholder/scenario if present.>\n"
        'E: "<copy ONE exact sentence from the chunk; include the final period>\"\n\n'
        f'Anchor phrase that must appear verbatim in Q: "{anchor}"\n'
        + template_hint +
        "Do not say “in the text/passage”. Do not invent facts.\n"
        "Select E as the single sentence that directly supports your answer.\n\n"
        f"Chunk:\n{chunk_text}"
    )


FEW_SHOT = ""

def build_qg_prompt(chunk_text: str, a) -> str:
    return (
        "You are writing study questions from the given chunk ONLY.\n"
        "Respond in English only.\n"
        f'Anchor phrase (must appear verbatim in the question): "{a}"\n\n'
        "Output exactly these three lines, in this order:\n"
        "Q: <one open question>\n"
        "A: <3–4 sentences covering: (1) the core idea; (2) one concrete detail from the chunk NOT used as E; "
        "(3) one implication/consequence or why it matters; (4) mention any named stakeholder/scenario.>\n"
        'E: "<copy ONE exact sentence from the chunk, include the final period>"\n\n'
        "Rules:\n"
        "• Start with How / Why / Explain / Compare / Under what conditions (avoid “What is…?” unless it’s a formal definition).\n"
        "• No ‘in the text/passage/chunk’. No invented facts. If no single sentence supports A, choose a different question.\n"
        f"• Make A **broad** (cover at least two distinct points from the chunk related to {a}) but still supported by E.\n\n"
        f"Chunk:\n{chunk_text}"
    )

def build_reg_prompt(chunk_text: str, a, q) -> str:
    return (
        "Use only this chunk.\n"
        "Produce ONE improved variant of the given question so that it is not definitional and remains open-ended.\n"
        "Vary the style (causal/comparative/conditional/mechanism) but keep the same anchor.\n"
        "Output exactly:\n"
        "Q: <one question>\n"
        "A: <3–4 sentences: core idea; one concrete detail from the chunk NOT used as E; a consequence/implication; stakeholder/scenario if present>\n"
        'E: "<one exact sentence from the chunk; include the final period>"\n'
        f"Anchor phrase (must appear verbatim in Q): {a}\n"
        "Do not write definition-style openings (e.g., “What is/are…”, “How is/are…”). Do not say “in the text/passage”.\n\n"
        f"Original question: {q}\n\n"
        f"Chunk:\n{chunk_text}"
    )



DEFY_RE = re.compile(r'^(?:how is|how are|what is|what are|define)\b', re.I)

def _answer_sentence_coverage(a: str, chunk_sents: list[str]) -> int:
    # count distinct chunk sentences that share ≥3 non-stopword tokens with A
    stop = set("""a an the and or of to in on for with as by from at is are was were be been being that this those these it its if then else when where while into over under across among between within without not no nor""".split())
    atok = [t for t in _tokens(a) if t not in stop]
    if not atok:
        return 0
    A = set(atok)
    count = 0
    for s in chunk_sents:
        S = set(t for t in _tokens(s) if t not in stop)
        if len(A & S) >= 3:
            count += 1
    return count

def score_qae_generic(q: str, a: str, e: str, chunk_raw: str) -> float:
    if not q or not a or not e:
        return -1e9
    # opener check (but no domain words)
    if not re.match(r'^(How|Why|Explain|Compare|Under what conditions)\b', q):
        return -5
    # avoid definition-style *patterns* (still domain-agnostic)
    if DEFY_RE.search(q):
        return -5
    # anchor presence will be checked outside (we only score content here)
    # E must be verbatim
    if e not in chunk_raw:
        return -5
    # A length: 3–5 sentences preferable
    sc = _sentence_count(a)
    len_score = 1.0 if 3 <= sc <= 5 else (0.5 if 2 <= sc <= 6 else -2)
    # coverage: reward referencing multiple chunk sentences
    sents = sent_tokenize(chunk_raw)
    cov = _answer_sentence_coverage(a, sents)
    cov_score = min(cov, 3) * 0.8
    # lexical diversity in Q
    qdiv = min(len(_tokens(q)), 12) * 0.05
    return len_score + cov_score + qdiv


def llm_generate_questions(chunk_text: str, anchor, q=None, temperature: float = 0.2) -> str:
    opts = {
        "temperature": 0.5 if q is None else 0.4,
        "top_p": 0.95,
        "num_predict": 384,
        "gpu_layers": 999,
        "num_ctx": 1024,
        "num_batch": 128,
    }
    if q:
        # regeneration path: keep your existing single-item prompt
        prompt = build_reg_prompt(chunk_text, anchor, q)
        resp = _safe_generate(model=LLM_MODEL, prompt=prompt, options=opts)
        return resp["response"]

    # multi-candidate path
    prompt = build_qg_prompt_variants(chunk_text, anchor)
    resp = _safe_generate(model=LLM_MODEL, prompt=prompt, options=opts)
    return resp["response"]


def best_evidence_for_answer(a: str, chunk_raw: str) -> str | None:
    sents = sent_tokenize(chunk_raw)
    stop = set("""a an the and or of to in on for with as by from at is are was were be been being that this those these it its if then else when where while into over under across among between within without not no nor""".split())
    A = set(t for t in _tokens(a) if t not in stop)
    best_s, best_score = None, 0.0
    for s in sents:
        S = set(t for t in _tokens(s) if t not in stop)
        if not S: 
            continue
        # score = overlap + slight length prior to prefer complete statements
        jac = len(A & S) / float(len(A | S)) if (A | S) else 0.0
        score = jac + min(len(S), 25) * 0.005
        if score > best_score:
            best_s, best_score = s, score
    return best_s



def get_qae_or_regen(chunk_raw: str, anchor_raw: str, prev_q: str | None = None):
    resp = llm_generate_questions(chunk_raw, anchor_raw, q=prev_q, temperature=0.2)
    items = parse_qae_items(resp) if not prev_q else [parse_qae(resp)]
    # filter candidates: must include anchor in Q and in E (normalized)
    anc_n = normalize_for_match(anchor_raw)
    filtered = []
    for it in items:
        if not it.get('q') or not it.get('e'):
            continue
        if anc_n not in normalize_for_match(it['q']):
            continue
        if anc_n not in normalize_for_match(it['e']):
            continue
        filtered.append(it)
    if not filtered:
        # one regen attempt with previous Q (if any), else fallback to your old regen
        if not prev_q and items and items[0].get('q'):
            return get_qae_or_regen(chunk_raw, anchor_raw, prev_q=items[0]['q'])
        return items[0] if items else {'ok': False, 'q': None, 'a': None, 'e': None, 'raw': resp, 'missing': set(['Q','A','E'])}

    # score and choose the best
    best = max(filtered, key=lambda it: score_qae_generic(it['q'], it['a'], it['e'], chunk_raw))
    best['ok'] = True
    return best



BAD_ANCHOR_PAT = re.compile(
    r"""(?ix)
    ^(?:figure|fig|table|chapter|section|page|pp?|http|www)\b
    |^\d{1,4}$
    |^\w{1,2}$
    |^[^A-Za-z]*$
    |(?:mcs\s*-\s*ftl|mcs\s*ftl)
    """
)
COMMON_STOP_PHRASES = set("""
abstract introduction conclusion references appendix acknowledgments copyright
figure fig table chapter section page pages url http https www et al
problem example exercise theorem lemma proof definition remark note
year years january february march april may june july august september october november december
""".split())

def _sentences(s: str) -> list[str]:
    return [x for x in re.split(r'(?<=[.!?])\s+', s) if x.strip()]

def _token_words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z0-9\-']+", s)

def _is_stop_phrase(s: str) -> bool:
    w = normalize_for_match(s)
    return w in COMMON_STOP_PHRASES or BAD_ANCHOR_PAT.search(w) is not None

def _score_anchor(s: str) -> float:
    score = 0.0
    s0 = s.strip()
    if 3 <= len(s0) <= 70 and re.search(r"[A-Za-z]", s0) and not BAD_ANCHOR_PAT.search(s0):
        score += 1.0
    if " " in s0:
        score += 0.3
    if re.search(r"[0-9/\\|]", s0):
        score -= 0.5
    return score

def extract_keywords_chunkwise(chunk_idx: int,
                               chunks: list[str],
                               df, per_chunk_tf, chunk_norms, chunk_sents, position_maps,
                               topn: int = 10) -> list[str]:
    N = len(chunks)
    ch = chunks[chunk_idx]
    tf = per_chunk_tf[chunk_idx]
    sents = chunk_sents[chunk_idx]
    pos_map = position_maps[chunk_idx]

    # score candidates
    scored = []
    for cand in tf.keys():
        base = tfidf_score(cand, tf, df, N)
        # bonus if appears in earlier sentences (likely a section/definition line)
        pos_bonus = 0.4 * (1.0 - (pos_map.get(cand, 0) / max(1, len(sents)-1)))
        # casing signal: title case often marks named concepts
        case_bonus = 0.3 if re.search(r"\b[A-Z][a-z]+\b", cand) else 0.0
        # length prior: prefer multi-word but cap
        tok = len(_token_words(cand))
        len_bonus = 0.15 * min(tok, 6)
        scored.append((cand, base + pos_bonus + case_bonus + len_bonus))

    scored.sort(key=lambda x: x[1], reverse=True)

    # diversity via MMR on token sets (Jaccard)
    cand_vecs = {c: _tokens(c) for c, _ in scored}
    ordered = [c for c, _ in scored]
    diverse = mmr_select(ordered, cand_vecs, k=min(topn, len(ordered)), diversity=0.65)
    return diverse

def pick_anchor(kw_list: list[str], chunk_raw: str) -> str:
    # choose the first whose normalized form appears in chunk (guard against spurious phrases)
    cn = normalize_for_match(chunk_raw)
    for cand in kw_list:
        if normalize_for_match(cand) in cn:
            return cand
    return kw_list[0] if kw_list else "key idea"


def _sentence_count(s: str) -> int:
    return len([x for x in re.split(r'[.!?](?:\s+|$)', s.strip()) if x.strip()])

def quality_ok(anchor_raw, q, a, e, chunk_raw) -> bool:
    if not q or not a or not e:
        return False
    if not re.match(r'^(How|Why|Explain|Compare|Under what conditions)\b', q):
        return False
    if normalize_for_match(anchor_raw) not in normalize_for_match(q):
        return False
    if normalize_for_match(anchor_raw) not in normalize_for_match(e):
        return False
    if normalize_for_match(e) not in normalize_for_match(chunk_raw):
        return False
    sc = _sentence_count(a)
    if sc < 2 or sc > 5:
        return False
    if BAD_ANCHOR_PAT.search(q.lower()):
        return False
    return True

def extract_q_e(text):
    q = text.split("Q:", 1)[1].split("\n", 1)[0].strip()
    e = text.split('E:', 1)[1].strip()
    # strip surrounding quotes if present
    if e.startswith('"') and '"' in e[1:]:
        e = e[1:e[1:].find('"')+1]  # keep inside quotes incl. closing quote
        e = e.strip('"').strip()
    return e, q
        

def validate_qs(chunk, keywords, evidence):
    ch = normalize_for_match(chunk)
    ev = normalize_for_match(evidence)
    kws = [normalize_for_match(kw) for kw in keywords]
    return ch, kws, ev
    

def check_evidence_chunk(chunk_norm, chunk_raw, anchor_norm, anchor_raw, e_raw, q_raw, a_raw, kw_8, regen_left=1):
    """
    1) Evidence must be verbatim substring of RAW chunk.
    2) If not, try one regen (parse with parse_qae) and re-check.
    3) If still bad and regen_left exhausted, return what we have.
    """
    # VERBATIM check uses RAW strings
    if e_raw and e_raw in chunk_raw:
        return check_anchor_evidence(chunk_norm, chunk_raw, anchor_norm, anchor_raw, e_raw, q_raw, a_raw, kw_8, regen_left)

    if regen_left <= 0:
        return q_raw, a_raw, e_raw

    # Regenerate with same anchor and previous question; PARSE Q/A/E
    resp = llm_generate_questions(chunk_raw, anchor_raw, q=q_raw, temperature=0.4)
    item = parse_qae(resp)
    if not item['ok']:
        return q_raw, a_raw, e_raw

    return check_evidence_chunk(
        chunk_norm, chunk_raw, anchor_norm, anchor_raw,
        item['e'], item['q'], item['a'], kw_8, regen_left=0
    )


def check_anchor_evidence(chunk_norm, chunk_raw, anchor_norm, anchor_raw, e_raw, q_raw, a_raw, kw_8, regen_left):
    """
    1) Anchor must appear (normalized) inside the evidence.
    2) If not, try one regen (parse) with same anchor.
    3) If still bad, try the next anchor from kw_8 (fresh Q/A/E), and re-validate evidence-in-chunk.
    """
       # ANCHOR-IN-EVIDENCE uses NORMALIZED strings
    if e_raw and anchor_norm in normalize_for_match(e_raw):
        return q_raw, a_raw, e_raw

    if regen_left > 0:
        resp = llm_generate_questions(chunk_raw, anchor_raw, q=q_raw, temperature=0.4)
        item = parse_qae(resp)
        if item['ok'] and anchor_norm in normalize_for_match(item['e']):
            return item['q'], item['a'], item['e']

    # Try next anchor
    try:
        i = kw_8.index(anchor_raw)
        next_anchor_raw = kw_8[i+1]
    except (ValueError, IndexError):
        return q_raw, a_raw, e_raw

    next_anchor_norm = normalize_for_match(next_anchor_raw)

    # Fresh question for the next anchor; PARSE Q/A/E
    resp = llm_generate_questions(chunk_raw, next_anchor_raw, q=None, temperature=0.2)
    item = parse_qae(resp)
    if not item['ok']:
        return q_raw, a_raw, e_raw

    # Ensure new evidence is actually in the chunk; if not, run evidence checker again
    return check_evidence_chunk(
        chunk_norm, chunk_raw, next_anchor_norm, next_anchor_raw,
        item['e'], item['q'], item['a'], kw_8, regen_left=1
    )

def _norm_q(q: str) -> str:
    # reuse your normalizer for consistency
    return normalize_for_match(q)

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z][A-Za-z0-9\-']+", s.lower()))

def _near_dupe(q: str, seen_norm_qs: set[str], seen_token_sets: list[set[str]], jaccard_threshold: float = 0.75) -> bool:
    nq = _norm_q(q)
    if nq in seen_norm_qs:
        return True
    qt = _tokens(q)
    if not qt:
        return False
    # lightweight near-duplicate check via Jaccard similarity on token sets
    for st in seen_token_sets:
        inter = len(qt & st)
        if inter == 0:
            continue
        jac = inter / float(len(qt | st))
        if jac >= jaccard_threshold:
            return True
    return False


OPENERS = ["How", "Why", "Explain", "Compare", "Under what conditions"]
_opener_idx = 0

def rotate_opener(q: str) -> str:
    global _opener_idx
    if re.match(r'^(How|Why|Explain|Compare|Under what conditions)\b', q):
        return q
    opener = OPENERS[_opener_idx % len(OPENERS)]
    _opener_idx += 1
    # If model started with e.g. "Discuss", replace first token with desired opener
    parts = q.split(None, 1)
    tail = parts[1] if len(parts) > 1 else ""
    return f"{opener} {tail}".strip()



def test_main():
    size = 3000 # per question
    mode = input_content()
    text = clean_text(extract_content(mode))

    print(f"[INFO] Extracted chars: {len(text)}")

    # Optional safe mode cap (protects from pathological PDFs)
    MAX_CHARS = 200_000
    if len(text) > MAX_CHARS:
        print(f"[WARN] Truncating text to {MAX_CHARS} chars for safety.")
        text = text[:MAX_CHARS]


    chunks = join_chunks(size, text, overlap_sents=2)

    # Hard cap to avoid runaway chunk counts on messy PDFs
    MAX_CHUNKS = 60
    if len(chunks) > MAX_CHUNKS:
        print(f"[WARN] Limiting chunks from {len(chunks)} to {MAX_CHUNKS}.")
        chunks = chunks[:MAX_CHUNKS]

    df, per_chunk_tf, chunk_norms, chunk_sents, position_maps = build_corpus_stats(chunks)

    print(f"[INFO] Num chunks: {len(chunks)} (chunk_size={size}, overlap=2)")

    chunk_counter = 0
    ans_bank = []
    emit_idx = 0

    # === outputs ===
    out_dir = "runs"
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = os.path.join(out_dir, f"qae-{stamp}.jsonl")
    csv_path   = os.path.join(out_dir, f"qae-{stamp}.csv")

    # create files on first write
    if not os.path.exists(jsonl_path):
        open(jsonl_path, "w").close()
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["q_idx", "chunk_idx", "anchor", "Q", "A", "E"])

    # === de-dupe state (kept in case you hook it up later) ===
    _seen_norm_qs: set[str] = set()
    _seen_token_sets: list[set[str]] = []

    # === coverage ===
    coverage = {
        "total_chars": len(text),
        "chunks_count": len(chunks),   # set AFTER chunks is created
        "anchors_attempted": 0,
        "qa_ok": 0,
        "qa_fallback": 0,
        "unique_evidence": set(),
    }

    for chunk_idx, chunk_raw in enumerate(chunks, start=1):
        chunk_counter = chunk_idx
        kw_list = extract_keywords_chunkwise(
        chunk_idx-1, chunks, df, per_chunk_tf, chunk_norms, chunk_sents, position_maps, topn=10
    )
        top_k = min(2, len(kw_list))   # try a couple anchors per chunk
        anchors_tried = 0

        for anchor_raw in kw_list[:top_k]:
            coverage["anchors_attempted"] += 1

            item = get_qae_or_regen(chunk_raw, anchor_raw, prev_q=None)
            q = item.get('q'); a = item.get('a'); evidence = item.get('e')

            if not q or not a or not evidence:
                q = f"How does {anchor_raw} relate to this chunk?"
                sents = re.split(r'(?<=[.!?])\s+', chunk_raw.strip())
                def _norm(s): return normalize_for_match(s)
                evidence = next(
                    (s for s in sents if _norm(anchor_raw) in _norm(s)),
                    (sents[0] if sents else chunk_raw.strip())
                )
                a = "Answer: see the evidence and surrounding context in the chunk."
                coverage["qa_fallback"] += 1

            # sanitize anchor and skip junky ones
            anchor_raw = clean_text(anchor_raw).strip().strip('",.- ')
            if (not re.search(r'[A-Za-z]', anchor_raw)) or len(anchor_raw) <= 2:
                # try to salvage from Q (take a mid-length noun-ish phrase)
                m = re.search(r'(?i)\b([A-Za-z][A-Za-z\- ]{3,40})\b', q or '')
                anchor_raw = (m.group(1).strip() if m else 'key idea')

            # ensure evidence is a clean, single sentence that ends properly
            if evidence:
                evidence = evidence.strip()
                if not re.search(r'[.!?]"?$', evidence):
                    evidence += '.'
                if not evidence or evidence not in chunk_raw:
                    ev2 = best_evidence_for_answer(a or "", chunk_raw)
                    if ev2:
                        evidence = ev2 if ev2.endswith(('.', '!', '?')) else (ev2 + '.')


            # verify evidence + anchor; may regenerate once
            chunk_norm, _, _ = validate_qs(chunk_raw, kw_list, evidence)
            anchor_norm = normalize_for_match(anchor_raw)
            q, a, evidence = check_evidence_chunk(
                chunk_norm=chunk_norm, chunk_raw=chunk_raw,
                anchor_norm=anchor_norm, anchor_raw=anchor_raw,
                e_raw=evidence, q_raw=q, a_raw=a, kw_8=kw_list, regen_left=1
            )

            q = rotate_opener(q)
            if _near_dupe(q, _seen_norm_qs, _seen_token_sets, jaccard_threshold=0.70):
                continue
            _seen_norm_qs.add(_norm_q(q))
            _seen_token_sets.append(_tokens(q))

            # coverage
            coverage["qa_ok"] += 1
            if evidence:
                coverage["unique_evidence"].add(evidence)

            # outputs
            ans_bank.append(a)
            anchors_tried += 1
            emit_idx += 1

            # print to console
            print(f"{chunk_counter}.{anchors_tried}")
            print(q)

            # append to files
            with open(jsonl_path, "a", encoding="utf-8") as jf:
                jf.write(json.dumps({
                    "q_idx": emit_idx,
                    "chunk_idx": chunk_counter,
                    "anchor": anchor_raw,
                    "Q": q, "A": a, "E": evidence
                }, ensure_ascii=False) + "\n")
            with open(csv_path, "a", newline="", encoding="utf-8") as cf:
                csv.writer(cf).writerow([emit_idx, chunk_counter, anchor_raw, q, a, evidence])

    # tiny run summary
    # --- build Quizlet-ready exports ---
    quizlet = input("Would you like to export Q/A for quizlet?: ").lower()
    if quizlet in ["y", "yes"]:
        quizlet_dir = out_dir
        qa_path = os.path.join(quizlet_dir, f"quizlet-qa-{stamp}.csv")   # Q -> A
        qe_path = os.path.join(quizlet_dir, f"quizlet-qe-{stamp}.csv")   # Q -> E

        # read back master CSV and write 2-col files
        rows = []
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                # light cleanup to avoid stray newlines/commas in fields
                Q = (row.get("Q") or "").replace("\n", " ").strip()
                A = (row.get("A") or "").replace("\n", " ").strip()
                E = (row.get("E") or "").replace("\n", " ").strip()
                rows.append((Q, A, E))

        # write Q->A
        with open(qa_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Question", "Answer"])
            for Q, A, _E in rows:
                if Q and A:
                    w.writerow([Q, A])

        # write Q->E (good for “locate the line that supports it” drills)
        with open(qe_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Question", "Answer"])
            for Q, _A, E in rows:
                if Q and E:
                    w.writerow([Q, E])

        print(f"[OK] Quizlet files written:\n - {qa_path}\n - {qe_path}")

    print("\n--- run summary ---")
    print(f"chars: {coverage['total_chars']}")
    print(f"chunks: {coverage['chunks_count']}")
    print(f"anchors attempted: {coverage['anchors_attempted']}")
    print(f"Q/A ok: {coverage['qa_ok']}  |  fallbacks: {coverage['qa_fallback']}")
    print(f"unique evidence sentences: {len(coverage['unique_evidence'])}")

    answer = input("Would you like to see answers? :").lower()
    if answer in ["y", "yes"]:
        for idx, ans in enumerate(ans_bank, start=1):
            print(f"{idx}) {ans}\n")


if __name__ == "__main__":
    # IMPORTANT for multiprocessing on macOS/Windows and for safety on Linux
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method already set – ignore
        pass

    test_main()
