import magic
import string
from ollama import Client
from rake_nltk import Rake
import fitz 
import pandas
import docx
import re


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


def extract_from_pdf(file_path):
    doc = fitz.open(file_path)
    return '\n'.join([page.get_text("text") for page in doc])


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

def build_qg_prompt(chunk_text: str, a) -> str:
    return (
        "Use only this chunk.\n"
        f'Ask ONE open question about the anchor phrase: "{a}".\n'
        "Output exactly these three lines, in this order:\n"
        "Q: <one question>\n"
        "A: <3–4 sentences that cover: (1) the core idea; (2) one concrete detail from the chunk that is NOT your E sentence; "
        "(3) a consequence/implication or why it matters; (4) if the chunk names a stakeholder or scenario, mention it. Do not copy the E sentence.>\n"
        'E: "<copy ONE exact sentence from the chunk; no paraphrase; include the final period>"\n'
        "Rules:\n"
        f"• The question must include the anchor phrase verbatim: {a}\n"
        "• Start with How / Why / Explain / Compare / Under what conditions (avoid generic “What is…?” forms).\n"
        "• Do NOT say “in the chunk/text/passage”.\n"
        "• Do not invent facts. If no single sentence supports A, choose a different question.\n\n"
        "In A, cover at least two distinct points from the chunk related to the anchor (no outside facts).\n\n"
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
    if q:
        prompt = build_reg_prompt(chunk_text, anchor, q) 
        resp = ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
            options={"temperature": 0.5, "top_p": 0.95, "num_predict": 140, "gpu_layers": 99, "num_ctx": 1024, "num_batch": 256, "kv_cache_type": "cpu"}
        )
        return resp["response"]

    else:
        prompt = build_qg_prompt(chunk_text, anchor)
        resp = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": 0.5, "top_p": 0.95, "num_predict": 140, "gpu_layers": 99, "num_ctx": 1024, "num_batch": 256, "kv_cache_type": "cpu"}
        )
        # Ollama returns a dict; the text is in 'response'
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


def extract_keywords(chunk):
    r = Rake()
    text = normalize_for_rake(chunk)
    r.extract_keywords_from_text(text)
    phrases = r.get_ranked_phrases()
    return phrases[:8]


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


def test_main():
    size = 1500 # per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)
    chunk_counter = 0
    ans_bank = []
    for chunk_raw in chunks:
        chunk_counter = chunk_counter + 1
        kw_8 = extract_keywords(chunk_raw)
        anchor_raw = kw_8[0]

        response = llm_generate_questions(chunk_raw, anchor_raw, q=None, temperature=0.2)

        item = parse_qae(response)
        if not item['ok']:
            # one regen if Q/A/E tags missing
            response = llm_generate_questions(chunk_raw, anchor_raw, q=item.get('q') or "", temperature=0.2)
            item = parse_qae(response)
            if not item['ok']:
                # fallback: synthesize minimally valid Q/A/E so numbering doesn't skip
                q_fallback = f"How does {anchor_raw} relate to this chunk?"
                # pick a sentence from the chunk as evidence (prefer one containing the anchor)
                sents = re.split(r'(?<=[.!?])\s+', chunk_raw.strip())
                def _norm(s): return normalize_for_match(s)
                e_fallback = next((s for s in sents if _norm(anchor_raw) in _norm(s)), sents[0] if sents else chunk_raw.strip())
                a_fallback = "Answer: see highlighted evidence and surrounding context in the chunk."
                item = {'ok': True, 'q': q_fallback, 'a': a_fallback, 'e': e_fallback, 'raw': '', 'missing': set()}


        q = item['q']
        a = item['a']                  
        evidence = item['e']

        chunk_norm, _, _ = validate_qs(chunk_raw, kw_8, evidence)  # ev not needed here
        anchor_norm = normalize_for_match(anchor_raw)

        q, a, evidence = check_evidence_chunk(chunk_norm=chunk_norm, chunk_raw=chunk_raw, anchor_norm=anchor_norm, anchor_raw=anchor_raw, e_raw=evidence, q_raw=q, a_raw=a, kw_8=kw_8, regen_left=1)


        ans_bank.append(a)
        print(chunk_counter)
        print(q)

    answer = input("Would you like to see answers? :").lower()
    if answer in ["y", "yes"]:
        counter = 0
        for ans in ans_bank:
            counter = counter + 1
            print(f"{counter}) {ans}\n")



test_main()

