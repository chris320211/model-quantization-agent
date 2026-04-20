from __future__ import annotations

import logging
import uuid
from typing import Iterable

import boto3
import requests
import yaml
import io

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import FieldCondition, Filter, MatchValue

from .config import load_settings
from .retrieval import COLLECTION, get_qdrant_client, get_vectorstore

log = logging.getLogger(__name__)


def _doc_id(source: str, chunk_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}:{chunk_idx}"))


def _already_indexed(source: str) -> bool:
    try:
        client = get_qdrant_client()
        results, _ = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return bool(results)
    except Exception:
        return False


def _get_r2_client():
    s = load_settings()
    return boto3.client(
        "s3",
        endpoint_url=f"https://{s.r2_account_id}.r2.cloudflarestorage.com",
        aws_access_key_id=s.r2_access_key_id,
        aws_secret_access_key=s.r2_secret_access_key,
    )


def _r2_key_exists(key: str) -> bool:
    try:
        s = load_settings()
        _get_r2_client().head_object(Bucket=s.r2_bucket_name, Key=key)
        return True
    except Exception:
        return False


def _upload_to_r2(key: str, data: bytes, content_type: str) -> None:
    s = load_settings()
    _get_r2_client().put_object(
        Bucket=s.r2_bucket_name,
        Key=key,
        Body=data,
        ContentType=content_type,
    )
    log.info("Uploaded to R2: %s (%d bytes)", key, len(data))


def _fetch_github_readme(repo_url: str, token: str | None) -> str | None:
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


def _download_repo_zip(repo_url: str, token: str | None) -> bytes | None:
    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
    except (IndexError, ValueError):
        return None
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    for branch in ("main", "master"):
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        try:
            r = requests.get(url, headers=headers, timeout=120)
            if r.status_code == 200:
                return r.content
        except requests.RequestException as e:
            log.warning("Repo zip download error for %s (%s): %s", repo_url, branch, e)
    log.warning("Repo zip download failed for %s", repo_url)
    return None


def _download_arxiv_pdf(arxiv_id: str) -> bytes | None:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            return r.content
        log.warning("arXiv PDF download failed for %s: %s", arxiv_id, r.status_code)
    except requests.RequestException as e:
        log.warning("arXiv PDF download error for %s: %s", arxiv_id, e)
    return None


def _fetch_arxiv(arxiv_id: str) -> list[Document]:
    try:
        import pypdf
        pdf_bytes = _download_arxiv_pdf(arxiv_id)
        if not pdf_bytes:
            return []
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"arxiv_id": arxiv_id})]
    except Exception as e:
        log.warning("Arxiv fetch failed for %s: %s", arxiv_id, e)
        return []


def _chunk(docs: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    return splitter.split_documents(list(docs))


def _method_summary_doc(method: dict) -> Document:
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
    """Build/update Qdrant index and R2 raw-file store from seed/methods.yaml."""
    s = load_settings()
    vs = get_vectorstore()

    with s.seed_path.open() as f:
        catalog = yaml.safe_load(f)

    added = 0
    for method in catalog:
        mid = method["id"]

        # 1. catalog card — always index
        card = _method_summary_doc(method)
        if not _already_indexed(card.metadata["source"]):
            vs.add_documents([card], ids=[_doc_id(card.metadata["source"], 0)])
            added += 1

        # 2. arxiv paper — embed text chunks + upload raw PDF to R2
        if method.get("arxiv_id"):
            arxiv_id = method["arxiv_id"]
            source = f"arxiv://{arxiv_id}"
            if not _already_indexed(source):
                docs = _fetch_arxiv(arxiv_id)
                for d in docs:
                    d.metadata.update({"source": source, "method_id": mid, "kind": "paper"})
                chunks = _chunk(docs)
                if chunks:
                    ids = [_doc_id(source, i) for i in range(len(chunks))]
                    vs.add_documents(chunks, ids=ids)
                    added += len(chunks)
                    log.info("Indexed arxiv %s: %d chunks", arxiv_id, len(chunks))

            r2_key = f"{mid}/paper_{arxiv_id}.pdf"
            if not _r2_key_exists(r2_key):
                pdf = _download_arxiv_pdf(arxiv_id)
                if pdf:
                    _upload_to_r2(r2_key, pdf, "application/pdf")

        # 3. each repo — embed README text chunks + upload full repo zip to R2
        for repo_url in method.get("repos", []) or []:
            parts = repo_url.rstrip("/").split("/")
            owner, repo = parts[-2], parts[-1]

            readme_source = f"github://{repo_url}"
            if not _already_indexed(readme_source):
                text = _fetch_github_readme(repo_url, s.github_token)
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": readme_source,
                            "method_id": mid,
                            "kind": "readme",
                            "repo": repo_url,
                        },
                    )
                    chunks = _chunk([doc])
                    if chunks:
                        ids = [_doc_id(readme_source, i) for i in range(len(chunks))]
                        vs.add_documents(chunks, ids=ids)
                        added += len(chunks)
                        log.info("Indexed README %s: %d chunks", repo_url, len(chunks))

            r2_key = f"{mid}/repo_{owner}_{repo}.zip"
            if not _r2_key_exists(r2_key):
                zip_data = _download_repo_zip(repo_url, s.github_token)
                if zip_data:
                    _upload_to_r2(r2_key, zip_data, "application/zip")

    return added


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    n = ingest_all()
    print(f"Ingest complete. Added {n} new chunks to Qdrant Cloud.")


if __name__ == "__main__":
    main()
