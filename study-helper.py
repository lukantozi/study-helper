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
LLM_MODEL   = "qwen2.5:3b-instruct"

ollama_client = Client(host=OLLAMA_HOST)
translator = str.maketrans('', '', string.punctuation)


def extract_from_txt(file_path):
    with open(file_path) as doc:
        text = doc.read()
    return text


def extract_from_pdf(file_path):
    doc = fitz.open(file_path)
    return '\n'.join([page.get_text() for page in doc])


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

def build_qg_prompt(chunk_text: str, n: int = 5) -> str:
    return (
        "Use only this chunk.\n"
        f"Produce {n} open questions. For each item output exactly:\n"
        "Q: …\n"
        "A (3–4 sentences): …\n"
        'EVIDENCE (copy exact sentence from the chunk, no paraphrase, no ellipses; include the final period; '
        'if a sentence starts with a marker like “(2)”, omit the marker but keep the sentence text): "…"\n'
        "RULES:\n"
        "• Do not invent facts.\n"
        "• Every number in A must also appear in EVIDENCE.\n"
        "• If no single sentence supports A, choose a different question.\n\n"
        "Chunk:\n"
        f"{chunk_text}"
    )


def llm_generate_questions(chunk_text: str, n: int = 5, temperature: float = 0.2) -> str:
    prompt = build_qg_prompt(chunk_text, n=n)
    resp = ollama_client.generate(
        model=LLM_MODEL,
        prompt=prompt,
        options={"temperature": temperature}
    )
    # Ollama returns a dict; the text is in 'response'
    return resp["response"]


def extract_keywords(chunk):
    r = Rake()
    kw = []
#    for chunk in chunks:
#        r.extract_keywords_from_text(chunk)
#        kw.append(r.get_ranked_phrases())
    r.extract_keywords_from_text(chunk)
    kw.append(r.get_ranked_phrases())
    return kw[0][0:10]
        

def validate_qs(chunk, keywords):
    ch = ' '.join(chunk.lower().translate(translator).replace("-", " ").split())
    kws = [' '.join(kw.lower().translate(translator).replace("-", " ").split()) for kw in keywords]
    return ch, kws
    

def test_main():
    size = 1000 # per question
    mode = input_content()
    text = extract_content(mode)
    chunks = join_chunks(size, text)
#    kw = extract_keywords(chunks)
#    print(kw)
    chunk0 = chunks[0]
    kw_10 = extract_keywords(chunk0)
#    compare_ch_kw(chunk0, kw_10)
#    qa_text = llm_generate_questions(chunk0, n=5, temperature=0.2)

#    print("\n=== Q&A (chunk 0) ===\n")
#    print(qa_text)
    print(chunk0)
    print(kw_10)
    ch, kw = validate_qs(chunk0, kw_10)
    print("----------------------")
    print("----------------------")
    print(ch)
    print(kw)

test_main()

