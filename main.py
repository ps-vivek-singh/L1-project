import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAQQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)

def extract_llm_content(res: dict) -> str:
    if not isinstance(res, dict):
        return ""
    if "error" in res:
        err = res["error"]
        err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise RuntimeError(f"LLM API Error: {err_msg}")
    choices = res.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is not None:
                return str(content)
    return ""


NSE_SYSTEM_PROMPT = """
You are an AI assistant specialized in NSE Corporate Announcement documents.

Your job is to answer questions only from the uploaded NSE document.

The document may contain:
- Company Name
- Reporting Period
- Key Officials
- Complaint Status
- RTA Confirmation
- Regulatory filings
- Shareholder information
- Other corporate disclosures

Rules:

1. Answer only using the retrieved document context.
2. Do not use outside knowledge.
3. If the answer is not present in the context, reply:
   "I couldn't find enough information in the uploaded document."
4. Never guess or make assumptions.
5. Keep answers concise and factual.
6. If possible, mention the source page or chunk.
"""


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks, pages = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, pages=pages, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc | dict) -> RAGUpsertResult:
        if isinstance(chunks_and_src, dict):
            chunks_and_src = RAGChunkAndSrc.model_validate(chunks_and_src)
        chunks = chunks_and_src.chunks
        pages = chunks_and_src.pages or [None] * len(chunks)
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [
            {
                "source": source_id,
                "text": chunks[i],
                "page": pages[i] if i < len(pages) else None,
                "chunk_index": i + 1,
            }
            for i in range(len(chunks))
        ]
        store = QdrantStorage()
        store.clear()  # Wipes previous PDF vectors so only the newly uploaded PDF is active
        store.upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    if isinstance(chunks_and_src, dict):
        chunks_and_src = RAGChunkAndSrc.model_validate(chunks_and_src)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    if isinstance(ingested, dict):
        ingested = RAGUpsertResult.model_validate(ingested)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], pages=found.get("pages", []), sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)
    if isinstance(found, dict):
        found = RAGSearchResult.model_validate(found)

    context_items = []
    ctx_pages = found.pages if found.pages else [None] * len(found.contexts)
    for i, (c, p) in enumerate(zip(found.contexts, ctx_pages)):
        page_str = f", Page {p}" if p is not None else ""
        context_items.append(f"[Chunk {i + 1}{page_str}]:\n{c}")
    context_block = "\n\n".join(context_items)

    user_content = (
        "Use the following retrieved NSE document context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )

    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        adapter = ai.openai.Adapter(
            auth_key=gemini_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            model="gemini-2.5-flash"
        )
    else:
        adapter = ai.openai.Adapter(
            auth_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o-mini"
        )

    res = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "max_tokens": 1024,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": NSE_SYSTEM_PROMPT.strip()},
                {"role": "user", "content": user_content}
            ]
        }
    )


    answer = extract_llm_content(res).strip()
    if not answer:
        answer = "Could not generate an answer due to an LLM provider error. Please check your API key."
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
