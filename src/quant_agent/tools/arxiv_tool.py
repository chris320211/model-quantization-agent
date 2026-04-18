from __future__ import annotations

import hashlib

from langchain_community.document_loaders import ArxivLoader
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..retrieval import get_vectorstore


def _doc_id(source: str, chunk_idx: int) -> str:
    h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{h}-{chunk_idx}"


@tool
def arxiv_fetch(arxiv_id: str) -> str:
    """Fetch a paper from arxiv by its ID (e.g. '2310.19102') and add it to the local index.

    Use this when a method is not in the curated catalog but the user mentions a paper,
    or when rag_search returned nothing relevant for a novel method.
    """
    vs = get_vectorstore()
    source = f"arxiv://{arxiv_id}"
    existing = vs.get(where={"source": source}, limit=1)
    if existing and existing.get("ids"):
        return f"Paper {arxiv_id} already indexed."

    try:
        docs = ArxivLoader(query=arxiv_id, load_max_docs=1).load()
    except Exception as e:
        return f"Arxiv fetch failed for {arxiv_id}: {e}"
    if not docs:
        return f"No paper found for arxiv id {arxiv_id}."

    for d in docs:
        d.metadata.update({"source": source, "method_id": f"adhoc:{arxiv_id}", "kind": "paper"})
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1200, chunk_overlap=150
    ).split_documents(docs)
    ids = [_doc_id(source, i) for i in range(len(chunks))]
    vs.add_documents(chunks, ids=ids)
    title = docs[0].metadata.get("Title", arxiv_id)
    return f"Indexed arxiv {arxiv_id} ('{title}'): {len(chunks)} chunks."
