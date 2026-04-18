from __future__ import annotations

import hashlib
import logging
from typing import Iterable

import requests
import yaml
from langchain_community.document_loaders import ArxivLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_settings
from .retrieval import get_vectorstore

log = logging.getLogger(__name__)


def _doc_id(source: str, chunk_idx: int) -> str:
    h = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
    return f"{h}-{chunk_idx}"


def _already_indexed(vs, source: str) -> bool:
    try:
        hit = vs.get(where={"source": source}, limit=1)
        return bool(hit and hit.get("ids"))
    except Exception:
        return False


def _fetch_github_readme(repo_url: str, token: str | None) -> str | None:
    # repo_url like https://github.com/org/name
    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
    except (IndexError, ValueError):
        return None
    api = f"https://api.github.com/repos/{owner}/{repo}/readme"
    headers = {"Accept": "application/vnd.github.raw"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = requests.get(api, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.text
        log.warning("GitHub README fetch failed for %s: %s", repo_url, r.status_code)
    except requests.RequestException as e:
        log.warning("GitHub fetch error for %s: %s", repo_url, e)
    return None


def _fetch_arxiv(arxiv_id: str) -> list[Document]:
    try:
        loader = ArxivLoader(query=arxiv_id, load_max_docs=1, load_all_available_meta=False)
        return loader.load()
    except Exception as e:
        log.warning("Arxiv fetch failed for %s: %s", arxiv_id, e)
        return []


def _chunk(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(list(docs))


def _method_summary_doc(method: dict) -> Document:
    # Dense catalog card — short and always indexed so queries hit even
    # when arxiv/github calls fail.
    lines = [
        f"Method: {method['name']} (id: {method['id']})",
        f"Bits: {method.get('bits')}",
        f"Quantizes: {method.get('quantizes')}",
        f"Needs calibration: {method.get('needs_calibration')}",
        f"Inference backends: {method.get('inference_backends')}",
        f"Notes: {method.get('notes', '')}",
        f"Arxiv: {method.get('arxiv_id')}",
        f"Repos: {method.get('repos')}",
    ]
    return Document(
        page_content="\n".join(lines),
        metadata={
            "source": f"catalog://{method['id']}",
            "method_id": method["id"],
            "kind": "catalog",
        },
    )


def ingest_all() -> int:
    """Build/update the Chroma index from seed/methods.yaml. Returns chunk count added."""
    s = load_settings()
    vs = get_vectorstore()

    with s.seed_path.open() as f:
        catalog = yaml.safe_load(f)

    added = 0
    for method in catalog:
        mid = method["id"]

        # 1. always index the catalog card (small, idempotent via source key)
        card = _method_summary_doc(method)
        if not _already_indexed(vs, card.metadata["source"]):
            vs.add_documents([card], ids=[_doc_id(card.metadata["source"], 0)])
            added += 1

        # 2. arxiv paper, if any
        if method.get("arxiv_id"):
            source = f"arxiv://{method['arxiv_id']}"
            if not _already_indexed(vs, source):
                docs = _fetch_arxiv(method["arxiv_id"])
                for d in docs:
                    d.metadata.update({"source": source, "method_id": mid, "kind": "paper"})
                chunks = _chunk(docs)
                if chunks:
                    ids = [_doc_id(source, i) for i in range(len(chunks))]
                    vs.add_documents(chunks, ids=ids)
                    added += len(chunks)
                    log.info("Indexed arxiv %s: %d chunks", method["arxiv_id"], len(chunks))

        # 3. each repo README
        for repo in method.get("repos", []) or []:
            source = f"github://{repo}"
            if _already_indexed(vs, source):
                continue
            text = _fetch_github_readme(repo, s.github_token)
            if not text:
                continue
            doc = Document(
                page_content=text,
                metadata={"source": source, "method_id": mid, "kind": "readme", "repo": repo},
            )
            chunks = _chunk([doc])
            if chunks:
                ids = [_doc_id(source, i) for i in range(len(chunks))]
                vs.add_documents(chunks, ids=ids)
                added += len(chunks)
                log.info("Indexed README %s: %d chunks", repo, len(chunks))

    return added


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = ingest_all()
    print(f"Ingest complete. Added {n} new chunks to Chroma at {load_settings().chroma_dir}.")


if __name__ == "__main__":
    main()
