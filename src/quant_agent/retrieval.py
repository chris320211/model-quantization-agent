from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import load_settings

COLLECTION = "quantization_corpus"


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    s = load_settings()
    return HuggingFaceEmbeddings(model_name=s.embed_model)


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    s = load_settings()
    s.chroma_dir.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(s.chroma_dir),
    )
