import magic
import string
from ollama import Client
from rake_nltk import Rake
import docx
import re
import json, csv, os, time
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
except Exception:
    _NLP = None



# change the model when switching pc/laptop
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"

ollama_client = Client(host=OLLAMA_HOST)
translator = str.maketrans('', '', string.punctuation)

EXTRA_DASHES = r"[–—−-]"  # en dash, em dash, minus, non-breaking hyphen


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


def extract_from_pdf(
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

    Returns a single string:
      [body text]\n\n--- FOOTNOTES ---\n[footnotes joined]
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

    doc = fitz.open(file_path)
    pages_data = []
    top_candidates = []
    bot_candidates = []

    for pno in range(len(doc)):
        page = doc[pno]
        width, height = page.rect.width, page.rect.height
        top_y = height * TOP_FRACTION
        bot_y = height * (1.0 - BOTTOM_FRACTION)

        d = page.get_text("dict")

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
            "lines": lines,
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

    for pno, pdata in enumerate(pages_data):
        width, height = pdata["size"]

        if detect_images:
            for img in pdata["page"].get_images(full=True):
                xref = img[0]
                try:
                    pix = pdata["page"].get_pixmap(xref=xref)
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

        lines = pdata["lines"]

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
            from statistics import median
            sizes = [sz for _t, sz in page_foot if sz > 0]
            cutoff = (median(sizes) * 0.8) if sizes else 8.0
            notes = [t for t, sz in page_foot if sz > 0 and sz <= cutoff]
            if notes:
                footnotes_all.extend([f"[p.{pno+1}] {n}" for n in notes])

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
        content = input("Please paste the content here: ")
        return content
    elif mode == 2:
        content = input("Please input the filepath here: ")

        try:
            content_type = magic.from_file(content, mime=True)
        except FileNotFoundError:
            return extract_content(mode)

        if content_type == "text/plain":
            return extract_from_txt(content)
        elif content_type == "application/pdf":
            return extract_from_pdf(content)
        elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return extract_from_docx(content)
        else:
            print("Accepted formats: pdf, docx, txt")
            return extract_content(mode)


def split_sentences(text):
    return re.split(r'(?<=[.!?]) +', text)


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


def join_chunks(chunk_size, text):
    sentences = split_sentences(text)
    return chunks(chunk_size, sentences)

FEW_SHOT = """\
Q: Explain how strong connectivity differs from weak connectivity in directed graphs and why the distinction matters.
A: Strong connectivity means every vertex can reach every other via directed paths, whereas weak connectivity only requires connectivity after ignoring edge directions. This distinction matters because algorithms that rely on reachability (like scheduling or routing) need guarantees in the actual direction of flow. For example, a graph may be weakly connected yet fail to deliver data from one node to another along directed edges. Recognizing strong components enables correct decomposition of problems.
E: "A directed graph G is said to be strongly connected if for every pair of nodes u, v ∈ V, there is a directed path from u to v (and vice-versa) in G."
"""

def build_qg_prompt(chunk_text: str, a) -> str:
    return (
        "You are writing study questions from the given chunk ONLY.\n"
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
        "Good example:\n"
        f"{FEW_SHOT}\n\n"
        f"Chunk:\n{chunk_text}"
    )

def build_reg_prompt(chunk_text: str, a, q) -> str:
    return (
        "Use only this chunk.\n"
        f'Regenerate this question about "{a}": {q}\n'
        "Output exactly:\n"
        "Q: <one question>\n"
        "A: <3–4 sentences that cover: (1) core idea; (2) one different concrete detail from the chunk (not your E sentence); "
        "(3) a consequence/implication; (4) stakeholder/scenario if present. Do not copy the E sentence.>\n"
        'E: "<one exact sentence from the chunk; include the final period>"\n'
        "Rules:\n"
        f"• Include the anchor phrase verbatim: {a}\n"
        "• No “mentioned in the chunk/text”. Prefer How/Why/Explain/Compare.\n"
        "• Do not invent facts. If no single sentence supports A, pick a different question.\n\n"
        "In A, cover at least two distinct points from the chunk related to the anchor (no outside facts).\n\n"
        f"Chunk:\n{chunk_text}"
    )


def llm_generate_questions(chunk_text: str, anchor, q=None, temperature: float = 0.2) -> str:
    opts = {
        "temperature": 0.4,
        "top_p": 0.95,
        "num_predict": 300,
        "gpu_layers": 99,   # offload as many as possible
        "num_ctx": 1024,    # match your service OLLAMA_CONTEXT_LENGTH
        "num_batch": 256,   # bigger batch -> faster on GPU
    }
    if q:
        prompt = build_reg_prompt(chunk_text, anchor, q)
    else:
        prompt = build_qg_prompt(chunk_text, anchor)

    resp = ollama_client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options=opts
    )
    return resp["response"]


def get_qae_or_regen(chunk_raw: str, anchor_raw: str, prev_q: str | None = None):
    # 1st attempt
    resp = llm_generate_questions(chunk_raw, anchor_raw, q=prev_q, temperature=0.2)
    item = parse_qae(resp)

    # If malformed (missing Q/A/E), try one regen
    if not item['ok']:
        resp = llm_generate_questions(chunk_raw, anchor_raw, q=(item.get('q') or prev_q or ""), temperature=0.2)
        item = parse_qae(resp)

    return item  # dict with keys: ok, q, a, e, missing


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

def extract_keywords(chunk: str) -> list[str]:
    c = re.sub(r"\s+", " ", chunk.strip())
    cand = []
    if _NLP:
        doc = _NLP(c)
        for nc in doc.noun_chunks:
            t = re.sub(EXTRA_DASHES, "-", nc.text.strip())
            cand.append(t)
    else:
        r = Rake()
        r.extract_keywords_from_text(normalize_for_rake(c))
        cand = r.get_ranked_phrases()

    seen, cleaned = set(), []
    for s in cand:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(s)

    scored = [(s, _score_anchor(s)) for s in cleaned]
    scored = [s for (s, sc) in scored if sc > 0]

    freq = {}
    low = normalize_for_match(chunk)
    for s in list(scored):
        n = normalize_for_match(s)
        freq[s] = low.count(n)
    scored.sort(key=lambda s: (freq.get(s, 0), len(s)), reverse=True)

    return scored[:8] if scored else ["directed graph", "strong connectivity", "tournament graph"]

def pick_anchor(kw_8: list[str], chunk_raw: str) -> str:
    if not chunk_raw.strip():
        return kw_8[0] if kw_8 else "key idea"

    low_chunk = normalize_for_match(chunk_raw)
    sents_norm = [normalize_for_match(s) for s in _sentences(chunk_raw)]

    def score_candidate(cand: str) -> float:
        if not cand or _is_stop_phrase(cand):
            return -1e9
        n_cand = normalize_for_match(cand)
        if not re.search(r"[a-z]", n_cand):
            return -1e9
        freq = low_chunk.count(n_cand)
        if freq == 0:
            return -1e9
        coverage = sum(1 for sn in sents_norm if n_cand in sn)
        tok_count = len(_token_words(cand))
        letter_ratio = len(re.findall(r"[A-Za-z]", cand)) / max(1, len(cand))
        has_digit = bool(re.search(r"\d", cand))
        multiword_bonus = 0.4 if tok_count >= 2 else 0.0
        length_bonus = min(tok_count, 6) * 0.1
        digit_penalty = 0.6 if has_digit else 0.0
        header_penalty = 0.8 if BAD_ANCHOR_PAT.search(cand) else 0.0
        stop_like = 0.5 if cand.lower() in COMMON_STOP_PHRASES else 0.0

        return (2.0 * freq) + (1.2 * coverage) + multiword_bonus + length_bonus \
               + (0.3 * letter_ratio) - digit_penalty - header_penalty - stop_like

    scored = [(cand, score_candidate(cand)) for cand in kw_8]
    scored = [x for x in scored if x[1] > -1e8]
    if scored:
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]

    words = _token_words(chunk_raw)
    grams = set()
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            cand = " ".join(words[i:i+n])
            if not _is_stop_phrase(cand):
                grams.add(cand)

    scored_fallback = [(g, score_candidate(g)) for g in grams]
    scored_fallback = [x for x in scored_fallback if x[1] > -1e8]
    if scored_fallback:
        scored_fallback.sort(key=lambda t: t[1], reverse=True)
        return scored_fallback[0][0]

    return kw_8[0] if kw_8 else "key idea"


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


def test_main():
    size = 1500 # per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)
    chunk_counter = 0
    ans_bank = []
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
            csv.writer(f).writerow(["chunk_idx", "anchor", "Q", "A", "E"])

    # === de-dupe state ===
    _seen_norm_qs: set[str] = set()
    _seen_token_sets: list[set[str]] = []
    for chunk_raw in chunks:
        chunk_counter += 1
        kw_8 = extract_keywords(chunk_raw)
        anchor_raw = pick_anchor(kw_8, chunk_raw)

        response = llm_generate_questions(chunk_raw, anchor_raw, q=None, temperature=0.2)
        item = parse_qae(response)

        if not item['ok']:
            response = llm_generate_questions(chunk_raw, anchor_raw, q=item.get('q') or "", temperature=0.2)
            item = parse_qae(response)
        if not item['ok']:
            continue  # still malformed → skip

        q, a, evidence = item['q'], item['a'], item['e']

        # evidence and anchor fix pass
        chunk_norm, _, _ = validate_qs(chunk_raw, kw_8, evidence)
        anchor_norm = normalize_for_match(anchor_raw)
        q, a, evidence = check_evidence_chunk(
            chunk_norm=chunk_norm, chunk_raw=chunk_raw,
            anchor_norm=anchor_norm, anchor_raw=anchor_raw,
            e_raw=evidence, q_raw=q, a_raw=a, kw_8=kw_8, regen_left=1
        )

        # final gate — actually SKIP if not good
        if not quality_ok(anchor_raw, q, a, evidence, chunk_raw):
            continue

# skip duplicate-ish questions
        if _near_dupe(q, _seen_norm_qs, _seen_token_sets, jaccard_threshold=0.75):
            continue
        _seen_norm_qs.add(_norm_q(q))
        _seen_token_sets.append(_tokens(q))

        # write to JSONL
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "chunk_idx": chunk_counter,
                "anchor": anchor_raw,
                "Q": q,
                "A": a,
                "E": evidence
            }, ensure_ascii=False) + "\n")

        # write to CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([chunk_counter, anchor_raw, q, a, evidence])

        # keep for optional on-screen answers
        ans_bank.append(a)

        # show the question index + text
        print(chunk_counter)
        print(q)



    answer = input("Would you like to see answers? :").lower()
    if answer in ["y", "yes"]:
        counter = 0
        for ans in ans_bank:
            counter = counter + 1
            print(f"{counter}) {ans}\n")



test_main()

