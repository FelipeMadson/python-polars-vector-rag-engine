import math
from typing import List, Tuple
from pydantic import BaseModel, Field

class Document(BaseModel):
    id: str
    title: str
    content: str
    embedding: List[float] = Field(default_factory=list)

class VectorRAGEngine:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        self.dimension = dimension
        self.documents: List[Document] = []

    def add_document(self, doc: Document) -> None:
        if len(doc.embedding) != self.dimension:
            raise ValueError(f"Embedding size {len(doc.embedding)} does not match engine dimension {self.dimension}")
        self.documents.append(doc)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return dot / (norm1 * norm2)

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        if len(query_embedding) != self.dimension:
            raise ValueError(f"Query embedding size {len(query_embedding)} must match dimension {self.dimension}")
        
        scored = []
        for doc in self.documents:
            score = self.cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, score))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
