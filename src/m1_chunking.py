"""M1 Chunking — stub for Lab 24 (copy full implementation from Day 18)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def load_documents(data_dir: str = "data") -> list[dict]:
    """Load policy documents from data directory."""
    import os, json
    docs = []
    if not os.path.isdir(data_dir):
        return docs
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                docs.append({"text": f.read(), "metadata": {"source": fname}})
        elif fname.endswith(".json"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    docs.extend(data)
                else:
                    docs.append(data)
    return docs


def chunk_hierarchical(text: str, metadata: dict | None = None,
                        parent_size: int = 2048, child_size: int = 256
                        ) -> tuple[list[Chunk], list[Chunk]]:
    """Split text into parent and child chunks."""
    metadata = metadata or {}
    words = text.split()
    parents, children = [], []
    for i in range(0, len(words), parent_size):
        parent_text = " ".join(words[i:i + parent_size])
        pid = f"p{i // parent_size}"
        parents.append(Chunk(text=parent_text, metadata={**metadata, "chunk_id": pid}))
        for j in range(0, len(parent_text.split()), child_size):
            child_text = " ".join(parent_text.split()[j:j + child_size])
            if child_text:
                children.append(Chunk(
                    text=child_text,
                    metadata={**metadata, "chunk_id": f"{pid}_c{j // child_size}"},
                    parent_id=pid,
                ))
    return parents, children
