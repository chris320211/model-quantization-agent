from __future__ import annotations

from langchain_core.tools import tool

from ..retrieval import get_vectorstore


@tool
def rag_search(query: str, k: int = 6) -> str:
    """Search the local quantization-literature index (arxiv papers + repo READMEs + method cards).

    Use this to ground recommendations in recent literature and to cite concrete sources.
    Returns the top-k chunks with their source URIs.
    """
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)
    if not results:
        return "No matches in the local index."
    out = []
    for i, d in enumerate(results, 1):
        src = d.metadata.get("source", "unknown")
        mid = d.metadata.get("method_id", "?")
        kind = d.metadata.get("kind", "?")
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 700:
            snippet = snippet[:700] + "…"
        out.append(f"[{i}] ({kind}, method={mid}) {src}\n{snippet}")
    return "\n\n".join(out)
