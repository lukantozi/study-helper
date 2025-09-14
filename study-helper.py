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
        "A: <3–4 sentences>\n"
        'E: "<copy ONE exact sentence from the chunk; no paraphrase; include the final period>"\n'
        "Rules: Do not invent facts. If no single sentence supports A, choose a different question.\n"
        "       Question has to prompt the student to answer with the 3-4 sentences (spaced repetition method).\n\n"
        f"Chunk:\n{chunk_text}"
    )


def build_reg_prompt(chunk_text: str, a, q) -> str:
    return(
        "Use only this chunk.\n"
        f'Regenerate this question about "{a}": {q}\n'
        "Output exactly:\n"
        "Q: <one question>\n"
        "A: <3–4 sentences>\n"
        'E: "<one exact sentence from the chunk; include the final period>"\n'
        "Reason: previous attempt failed formatting/verification.\n\n"
        "Rules: Do not invent facts. If no single sentence supports A, choose a different question.\n"
        "       Question has to prompt the student to answer with the 3-4 sentences (spaced repetition method).\n\n"
        f"Chunk:\n{chunk_text}"
    )


def llm_generate_questions(chunk_text: str, anchor, q=None, temperature: float = 0.2) -> str:
    if q:
        prompt = build_reg_prompt(chunk_text, anchor, q) 
        resp = ollama_client.generate(
                model=LLM_MODEL,
                prompt=prompt,
                options={"temperature": temperature}
        )
        return resp["response"]

    else:
        prompt = build_qg_prompt(chunk_text, anchor)
        resp = ollama_client.generate(
            model=LLM_MODEL,
            prompt=prompt,
            options={"temperature": temperature}
        )
        # Ollama returns a dict; the text is in 'response'
        return resp["response"]


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
    

def check_evidence_chunk(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8):
    match = chunk_norm.find(ev)
    print(match)
    if match != -1:
        checked = False
        return check_anchor_evidence(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8, checked)
    else:
        response = llm_generate_questions(chunk_raw, anchor_raw, q=q, temperature=0.4) # trying higher temperature for regeneration
        ev, q = extract_q_e(response)
        ev = normalize_for_match(ev)
        check_evidence_chunk(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8)


def check_anchor_evidence(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8, checked): 
    if checked:
        ind = kw_8.index(anchor_raw)
        anchor_norm = normalize_for_match(kw_8[ind+1])
        checked = False
    match = ev.find(anchor_norm)
    print(match)
    if match != -1:
        return q
    else:
        response = llm_generate_questions(chunk_raw, anchor_raw, q, temperature=0.4) # trying higher temperature for regeneration
        checked = True
        ev, q = extract_q_e(response)
        ev = normalize_for_match(ev)
        check_anchor_evidence(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8, checked)


def regenerate_check(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8):
    checked_text = check_evidence_chunk(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8) # None or response
    if checked_text:
        evidence, q = extract_q_e(checked_text)
        ev = normalize_for_match(evidence)

    return q

def test_main():
    size = 500 # per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)

    for chunk_raw in chunks:
        kw_8 = extract_keywords(chunk_raw)
        anchor_raw = kw_8[0]

        response = llm_generate_questions(chunk_raw, anchor_raw, q=None, temperature=0.2)
        evidence, q = extract_q_e(response)

        chunk_norm, _, ev = validate_qs(chunk_raw, kw_8, evidence)
        anchor_norm = kw_8[0]

        q = check_evidence_chunk(chunk_norm, chunk_raw, anchor_norm, anchor_raw, ev, q, kw_8)

        print(q)

test_main()

def main():
    size = 500# per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)
    for chunk in chunks:
        kw_8 = extract_keywords(chunk)
        anchor = kw_8[0]
        qe_text = llm_generate_questions(chunk, anchor, q=None, temperature=0.2)
        evidence, q = extract_q_e(qe_text)

