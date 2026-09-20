from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from bench import BENCHMARK_QUERIES, DATA_DIR, parse_frontmatter
from src.chunking import (
    FixedSizeChunker,
    MarkdownHeadingChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from src.embeddings import _mock_embed
from src.models import Document
from src.store import EmbeddingStore

chunkers = {
    "MarkdownHeadingChunker": MarkdownHeadingChunker(max_chunk_size=400),
    "RecursiveChunker": RecursiveChunker(chunk_size=400),
    "SentenceChunker": SentenceChunker(max_sentences_per_chunk=3),
    "FixedSizeChunker": FixedSizeChunker(chunk_size=400, overlap=40),
}

md_files = sorted(DATA_DIR.glob("*.md"))

# 1. Documents and chunks
documents_data = []
all_chunks_by_strategy = {c: [] for c in chunkers}

for p in md_files:
    fm, body = parse_frontmatter(p)
    doc_info = {
        "doc_id": p.stem,
        "filename": p.name,
        "metadata": fm,
        "body": body,
        "char_count": len(body),
        "chunks": {},
    }
    for cname, cinstance in chunkers.items():
        chunks = cinstance.chunk(body)
        chunk_objects = []
        for i, c in enumerate(chunks):
            cid = f"{p.stem}#{i}"
            vec = _mock_embed(c)
            chunk_obj = {
                "id": cid,
                "doc_id": p.stem,
                "index": i,
                "content": c,
                "length": len(c),
                "embedding": vec,
                "metadata": {**fm, "doc_id": p.stem},
            }
            chunk_objects.append(chunk_obj)
            all_chunks_by_strategy[cname].append(chunk_obj)
        doc_info["chunks"][cname] = chunk_objects
    documents_data.append(doc_info)

# 2. Benchmark Precomputed Results for all strategies
benchmark_results = {}
for cname, chunk_list in all_chunks_by_strategy.items():
    benchmark_results[cname] = {}
    # Create store for this strategy
    store = EmbeddingStore(embedding_fn=_mock_embed)
    docs = [Document(id=c["id"], content=c["content"], metadata=c["metadata"]) for c in chunk_list]
    store.add_documents(docs)

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        # Normal search with filter
        res = store.search_with_filter(q["query"], top_k=5, metadata_filter=q["filter"])
        # No filter search for A/B comparison
        res_nofilter = store.search_with_filter(q["query"], top_k=5, metadata_filter=None)
        benchmark_results[cname][qid] = {
            "with_filter": res,
            "no_filter": res_nofilter,
        }

# 3. Prediction pairs from REPORT_CANHAN
prediction_pairs = [
    {
        "pair_id": 1,
        "sentence_a": "The buyer requested a full refund for the damaged item.",
        "sentence_b": "The customer asked to return the broken product and get money back.",
        "prediction": "CAO",
        "mock_score": round(compute_similarity(_mock_embed("The buyer requested a full refund for the damaged item."), _mock_embed("The customer asked to return the broken product and get money back.")), 4),
        "note": "Cùng nghĩa đổi trả nhưng MockEmbedder dùng MD5 băm ký tự nên cho điểm âm!",
    },
    {
        "pair_id": 2,
        "sentence_a": "A seller can deduct up to 50% from the refund if returned used.",
        "sentence_b": "Eligible sellers may withhold half of the return amount for altered goods.",
        "prediction": "CAO",
        "mock_score": round(compute_similarity(_mock_embed("A seller can deduct up to 50% from the refund if returned used."), _mock_embed("Eligible sellers may withhold half of the return amount for altered goods.")), 4),
        "note": "Cùng nghĩa khấu trừ 50% nhưng từ vựng khác biệt.",
    },
    {
        "pair_id": 3,
        "sentence_a": "How to cancel an order on eBay after checkout.",
        "sentence_b": "Steps to ask the seller to cancel a purchase before shipment.",
        "prediction": "CAO",
        "mock_score": round(compute_similarity(_mock_embed("How to cancel an order on eBay after checkout."), _mock_embed("Steps to ask the seller to cancel a purchase before shipment.")), 4),
        "note": "Hỏi về quy trình hủy đơn mua.",
    },
    {
        "pair_id": 4,
        "sentence_a": "The seller must respond to a return request within 3 business days.",
        "sentence_b": "Photosynthesis is the process by which green plants convert light into chemical energy.",
        "prediction": "THẤP",
        "mock_score": round(compute_similarity(_mock_embed("The seller must respond to a return request within 3 business days."), _mock_embed("Photosynthesis is the process by which green plants convert light into chemical energy.")), 4),
        "note": "Hai chủ đề hoàn toàn tách biệt (TMĐT vs Sinh học).",
    },
    {
        "pair_id": 5,
        "sentence_a": "Can a buyer return an item if seller stated no returns?",
        "sentence_b": "Quantum entanglement is a phenomenon in quantum physics where particles become interconnected.",
        "prediction": "THẤP",
        "mock_score": round(compute_similarity(_mock_embed("Can a buyer return an item if seller stated no returns?"), _mock_embed("Quantum entanglement is a phenomenon in quantum physics where particles become interconnected.")), 4),
        "note": "Hai chủ đề hoàn toàn tách biệt (Chính sách eBay vs Vật lý lượng tử).",
    },
]

# Output to demo_data.js
output_js = f"""// AUTO-GENERATED DEMO DATA FOR LAB 7 UI
window.LAB7_DATA = {{
    documents: {json.dumps(documents_data, ensure_ascii=False, indent=2)},
    queries: {json.dumps(BENCHMARK_QUERIES, ensure_ascii=False, indent=2)},
    benchmark_results: {json.dumps(benchmark_results, ensure_ascii=False, indent=2)},
    prediction_pairs: {json.dumps(prediction_pairs, ensure_ascii=False, indent=2)}
}};
"""

out_path = ROOT_DIR / "demo_data.js"
out_path.write_text(output_js, encoding="utf-8")
print(f"Generated {out_path} ({len(output_js):,} bytes)")
