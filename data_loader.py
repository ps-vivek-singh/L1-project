import os
import requests
from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv

load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if not gemini_key:
    client = OpenAI()
    EMBED_MODEL = "text-embedding-3-large"

EMBED_DIM = 3072
splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    if gemini_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:batchEmbedContents?key={gemini_key}"
        payload = {
            "requests": [
                {
                    "model": "models/gemini-embedding-2",
                    "content": {
                        "parts": [{"text": t}]
                    }
                } for t in texts
            ]
        }
        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
        return [emb["values"] for emb in data.get("embeddings", [])]
    else:
        response = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts,
        )
        return [item.embedding for item in response.data]