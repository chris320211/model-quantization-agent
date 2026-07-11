"""Pinned baseline used by generated method and fp16-reference environments."""

RUNTIME_PACKAGES: tuple[str, ...] = (
    "transformers==4.46.3",
    "accelerate==1.1.1",
    "safetensors==0.4.5",
    "sentencepiece==0.2.0",
    "datasets==3.1.0",
)
