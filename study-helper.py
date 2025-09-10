import magic
import string
from ollama import Client
from rake_nltk import Rake
import fitz 
import pandas
import docx
import re


# change the model when using on pc
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL   = "qwen2.5:7b-instruct"

ollama_client = Client(host=OLLAMA_HOST)
translator = str.maketrans('', '', string.punctuation)

EXTRA_DASHES = r"[–—−-]"  # en dash, em dash, minus, non-breaking hyphen


def normalize_for_rake(s: str) -> str:
    s = re.sub(EXTRA_DASHES, "-", s)
    s = s.replace("\u00a0", " ")  # nbsp -> space
    s = s.replace("-", " ")
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

def build_qg_prompt(chunk_text: str, a, b, n: int = 2) -> str:
    return (
        "Use only this chunk.\n"
        f"Produce {n} open questions. For each item output exactly:\n"
        "(Questions should prompt 3-4 sentence answer) Format -> Q(question number): …\n"
        f"• Q1 about {a}."
        f"• Q2 about {b}."
        "EVIDENCE (copy exact sentence from the chunk, no paraphrase, no ellipses; include the final period; Format -> E:"
        'if a sentence starts with a marker like “(2)”, omit the marker but keep the sentence text): "…"\n'
        "RULES:\n"
        "• Do not invent facts.\n"
        "• If no single sentence supports A, choose a different question.\n\n"
        "Chunk:\n"
        f"{chunk_text}"
    )

#     "(Answer with 3-4 sentences) Format -> A: …\n"

def llm_generate_questions(chunk_text: str, anchor_a, anchor_b, q=None, n: int = 2, temperature: float = 0.2) -> str:
    if q:
        resp = ollama_client.generate(
                model=LLM_MODEL,
                prompt=f"Regenerate the question: {q}, with the same rules before",
                options={"temperature": temperature}
        )
        return resp["response"]

    prompt = build_qg_prompt(chunk_text, n=n, a=anchor_a, b=anchor_b)
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
        

def validate_qs(chunk, keywords, evidence):
    ch = normalize_for_match(chunk)
    ev = normalize_for_match(evidence)
    kws = [normalize_for_match(kw) for kw in keywords]
    return ch, kws, ev
    

def check_match(text, subtext, q): 
    if isinstance(subtext, list):
        for sub in subtext:
            match = text.find(sub)
            print(match)
            if match != -1:
                print(f"{sub} ---- {match}")
                break
        # generate the question again

    else:
        match = text.find(subtext)
        print(match)
        print(text)
        if match != -1:
            print(f"{subtext} ---- {match}")
            return
        else: 
            # generate the question again
            pass


def extract_q_e(text):
    evidence = text.split("E: ",1)[1]
    question = (text.split("Q1: ")[1]).split("E: ")[0]
    return evidence, question

def test_main():
    size = 1000 # per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)
    chunk0 = chunks[0]
    kw_8 = extract_keywords(chunk0)
    anchor_A = kw_8[0]
    anchor_B = kw_8[1]
    qe_text = llm_generate_questions(chunk0, anchor_A, anchor_B, n=1, temperature=0.2)
    evidence, q = extract_q_e(qe_text)
    print(q)
    ch, kw, ev = validate_qs(chunk0, kw_8, evidence)
    print("----------------------")
    print("-----chunk------------")
    print("----------------------")
    print(ch)
    print("----------------------")
    print("-----keywords---------")
    print("----------------------")
    print(kw)
    print("----------------------")
    print("--generated text------")
    print("----------------------")
    print(qe_text)
    print("----------------------")
    print("--evidence in chunk---")
    print("----------------------")
    check_match(ch, ev, q)
    print("----------------------")
    print("---keywords in ev-----")
    print("----------------------")
    check_match(ev, kw, q)


test_main()
   
