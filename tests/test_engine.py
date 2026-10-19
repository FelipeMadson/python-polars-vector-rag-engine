import pytest
from src.rag_engine.engine import VectorRAGEngine, Document

def test_engine_initialization_and_query():
    engine = VectorRAGEngine(dimension=3)
    
    doc1 = Document(
        id="doc-1",
        title="Distributed Systems",
        content="Overview of consensus protocols.",
        embedding=[1.0, 0.0, 0.0]
    )
    doc2 = Document(
        id="doc-2",
        title="Database Indexes",
        content="B-trees and LSM-trees comparison.",
        embedding=[0.0, 1.0, 0.0]
    )
    
    engine.add_document(doc1)
    engine.add_document(doc2)
    
    # Query aligned with doc1
    results = engine.query([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    best_doc, score = results[0]
    assert best_doc.id == "doc-1"
    assert pytest.approx(score, 0.001) == 1.0

def test_dimension_mismatch_raises_error():
    engine = VectorRAGEngine(dimension=4)
    doc = Document(id="d", title="t", content="c", embedding=[1.0, 2.0])
    with pytest.raises(ValueError, match="does not match engine dimension"):
        engine.add_document(doc)
