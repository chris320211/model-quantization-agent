from __future__ import annotations

from functools import lru_cache

from langchain_qdrant import QdrantVectorStore
from langchain_voyageai import VoyageAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import load_settings

COLLECTION = "quantization_corpus"
_VOYAGE_DIMS = 512  # voyage-3-lite output dimensions


@lru_cache(maxsize=1)
def get_embeddings() -> VoyageAIEmbeddings:
    s = load_settings()
    return VoyageAIEmbeddings(voyage_api_key=s.voyage_api_key, model="voyage-3-lite")


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    s = load_settings()
    return QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)


@lru_cache(maxsize=1)
def get_vectorstore() -> QdrantVectorStore:
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=_VOYAGE_DIMS, distance=Distance.COSINE),
        )
    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION,
        embedding=get_embeddings(),
    )
