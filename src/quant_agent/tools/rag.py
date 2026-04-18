from __future__ import annotations

from collections import defaultdict

from langchain_core.tools import tool

from ..retrieval import get_vectorstore


def _format_hit(i: int, d) -> str:
    src = d.metadata.get("source", "unknown")
    mid = d.metadata.get("method_id", "?")
    kind = d.metadata.get("kind", "?")
    snippet = d.page_content.strip().replace("\n", " ")
    if len(snippet) > 700:
        snippet = snippet[:700] + "…"
    return f"[{i}] ({kind}, method={mid}) {src}\n{snippet}"


@tool
def rag_search(query: str, k: int = 6, method_id: str | None = None) -> str:
    """Search the local quantization-literature index (arxiv papers + repo READMEs + method cards).

    Use this to ground recommendations in recent literature and to cite concrete sources.
    When `method_id` is set, restricts results to chunks tagged for that method in the catalog
    (e.g. 'awq', 'gptq', 'hqq'). Returns the top-k chunks with their source URIs.
    """
    vs = get_vectorstore()
    kwargs = {"k": k}
    if method_id:
        kwargs["filter"] = {"method_id": method_id}
    results = vs.similarity_search(query, **kwargs)
    if not results:
        return "No matches in the local index."
    return "\n\n".join(_format_hit(i, d) for i, d in enumerate(results, 1))


def rag_survey(query: str, k: int = 20) -> str:
    """Return a broad k-chunk survey grouped by method_id.

    Non-tool helper used by the Research agent to eyeball the catalog's coverage before
    picking 3-8 candidate methods. Not registered as a LangChain tool on purpose — the
    Research agent is structured-output, not a ReAct loop.
    """
    vs = get_vectorstore()
    results = vs.similarity_search(query, k=k)
    if not results:
        return "No matches in the local index."
    grouped: dict[str, list[str]] = defaultdict(list)
    for i, d in enumerate(results, 1):
        mid = d.metadata.get("method_id", "unknown")
        grouped[mid].append(_format_hit(i, d))
    sections = []
    for mid in sorted(grouped):
        sections.append(f"=== method_id={mid} ({len(grouped[mid])} chunks) ===")
        sections.extend(grouped[mid])
    return "\n\n".join(sections)
