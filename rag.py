import os
import io
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv
import re

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Check https://console.groq.com/docs/models if this model name is rejected
MODEL = "openai/gpt-oss-120b"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma = chromadb.Client()  # in-memory: resets when the app restarts
collection = chroma.get_or_create_collection("papers")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_pages(pdf_bytes):
    """Return a list of {page, text, ocr} dicts. Uses OCR if a page has no real text."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        used_ocr = False
        if len(text) < 50:  # probably a scanned page
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img).strip()
            used_ocr = True
        pages.append({"page": i + 1, "text": text, "ocr": used_ocr})
    return pages


def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping pieces of about `size` characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def index_pdf(pdf_bytes, filename):
    """Extract, chunk, embed and store a PDF. Returns (pages, chunks, ocr_pages)."""
    pages = extract_pages(pdf_bytes)
    docs, metas, ids = [], [], []
    for p in pages:
        for j, chunk in enumerate(chunk_text(p["text"])):
            if not chunk.strip():
                continue
            docs.append(chunk)
            metas.append({"source": filename, "page": p["page"]})
            ids.append(f"{filename}-p{p['page']}-c{j}")
    if docs:
        embeddings = embedder.encode(docs).tolist()
        collection.upsert(documents=docs, embeddings=embeddings,
                          metadatas=metas, ids=ids)
    return len(pages), len(docs), sum(p["ocr"] for p in pages)


def answer(question, k=6, max_distance=1.5):
    """Return (answer_text, [(source, page), ...])."""
    if collection.count() == 0:
        return "Please upload a paper first.", []

    q_emb = embedder.encode([question]).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)

    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        # print(f"distance={dist:.2f} | {meta['source']} p.{meta['page']}")  # for tuning
        # print("   TEXT:", doc[:200].replace("\n", " "))
        if dist <= max_distance:
            hits.append((doc, meta))

    if not hits:
        return "I couldn't find this in the uploaded papers.", []

    context = "\n\n".join(
        f"[{m['source']}, page {m['page']}]\n{d}" for d, m in hits
    )
    system_prompt = (
        "You are a research paper assistant. Answer ONLY using the provided context. "
        "If the context does not contain the answer, say you don't know. "
        "Mention the source and page for each fact you use."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    resp = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer_text = resp.choices[0].message.content or ""
    if answer_text.strip().lower().startswith(("i don't know", "i don’t know")):
        return answer_text, []
    sources = sorted({(m["source"], m["page"]) for _, m in hits})
    cited = [
        (s, p) for s, p in sources
        if s in answer_text and re.search(rf"\b{p}\b", answer_text)
    ]
    return answer_text, (cited or sources)
    