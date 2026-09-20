from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Fix Windows cp1252 output encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.agent import KnowledgeBaseAgent
from src.chunking import (
    FixedSizeChunker,
    MarkdownHeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    GeminiEmbedder,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

# ==============================================================================
# CHIẾN LƯỢC CHUNKING CÁ NHÂN (Mỗi thành viên chỉ đổi 1 dòng này để benchmark)
# Nguyễn Viết Đức (R3): MarkdownHeadingChunker
# ==============================================================================
CHUNKER_NAME = "MarkdownHeadingChunker"
CHUNKER = MarkdownHeadingChunker(max_chunk_size=400)

# Các chiến lược khác của nhóm để đối chiếu:
# CHUNKER_NAME = "RecursiveChunker"; CHUNKER = RecursiveChunker(chunk_size=400)
# CHUNKER_NAME = "SentenceChunker"; CHUNKER = SentenceChunker(max_sentences_per_chunk=3)
# CHUNKER_NAME = "FixedSizeChunker"; CHUNKER = FixedSizeChunker(chunk_size=400, overlap=40)

# Thư mục dữ liệu chính thức
DATA_DIR = Path("data/ebay-policies")

# 5 Benchmark Queries chung của nhóm (R2 chủ trì)
BENCHMARK_QUERIES = [
    {
        "id": 1,
        "type": "Tra cứu số liệu (Numerical)",
        "query": "How many business days does a seller have to respond to a return or item not received request before eBay can step in?",
        "gold_answer": "3 business days (If after 3 business days the issue isn't resolved, you or the buyer can ask eBay to step in).",
        "gold_doc_id": "ebay-seller-ask-ebay-step-in",
        "expected_substring": "3 business days",
        "filter": None,
    },
    {
        "id": 2,
        "type": "Hỏi điều kiện & Lọc đối tượng (Condition with metadata_filter)",
        "query": "Can an order return be requested if the listing specifies that returns are not accepted?",
        "gold_answer": "Yes, under eBay Money Back Guarantee, if an item arrives damaged, is faulty, or doesn't match the listing, the buyer can return it for a refund even if the seller doesn't accept returns.",
        "gold_doc_id": "ebay-buyer-returns-refunds",
        "expected_substring": "even if the seller doesn't accept returns",
        "filter": {"audience": "buyer"},
    },
    {
        "id": 3,
        "type": "Hỏi điều kiện & Khấu trừ hoàn tiền (Condition & Seller Protection)",
        "query": "What percentage can be deducted from a refund if an item is returned damaged or used?",
        "gold_answer": "You can deduct up to 50% from the refund to recover the lost value of the item, provided the refund is sent within 2 business days and eBay hasn't stepped in.",
        "gold_doc_id": "ebay-seller-protections",
        "expected_substring": "up to 50%",
        "filter": {"audience": "seller"},
    },
    {
        "id": 4,
        "type": "Hỏi quy trình (Procedure / Steps)",
        "query": "What are the steps for a buyer to report that an item has not arrived through My eBay?",
        "gold_answer": "Go to Purchases, find the item, select More actions beside the item, select 'I didn't receive it', check refund/item preference, enter details, and select Send request.",
        "gold_doc_id": "ebay-buyer-item-not-received",
        "expected_substring": "I didn't receive it",
        "filter": {"audience": "buyer"},
    },
    {
        "id": 5,
        "type": "Liệt kê chính sách (Listing / Options)",
        "query": "What are the return policy duration options that a seller can choose to offer on eBay?",
        "gold_answer": "No returns, 30-day buyer-paid returns, 30-day free returns, 60-day buyer-paid returns, 60-day free returns (and 14-day returns in categories like Camera Drones, Digital Cameras, Jewelry).",
        "gold_doc_id": "ebay-seller-return-policy",
        "expected_substring": "30-day",
        "filter": {"audience": "seller"},
    },
]


def parse_frontmatter(path: Path) -> tuple[dict, str]:
    """Tách frontmatter thành metadata và phần thân thành content."""
    text = path.read_text(encoding="utf-8")
    parts = text.split("---", 2)
    if len(parts) >= 3:
        fm_text = parts[1]
        body = parts[2].strip()
        metadata = {}
        for line in fm_text.splitlines():
            m = re.match(r"^(\w+):\s*(.+)$", line.strip())
            if m:
                val = m.group(2).strip().strip('"').strip("'")
                metadata[m.group(1)] = val
        return metadata, body
    return {}, text.strip()


def get_embedder():
    """Lấy embedding model phù hợp, có cache nếu dùng OpenAI."""
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder()
        except Exception as e:
            print(f"[Warning] LocalEmbedder fallback to MockEmbedder: {e}")
            return _mock_embed
    elif provider == "openai":
        try:
            base_embedder = OpenAIEmbedder()
            cache_file = Path(".embedding_cache_openai.json")
            cache = {}
            if cache_file.exists():
                try:
                    cache = json.loads(cache_file.read_text(encoding="utf-8"))
                except Exception:
                    cache = {}

            def cached_embed(text: str) -> list[float]:
                thash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if thash in cache:
                    return cache[thash]
                vec = base_embedder(text)
                cache[thash] = vec
                cache_file.write_text(json.dumps(cache), encoding="utf-8")
                return vec

            return cached_embed
        except Exception as e:
            print(f"[Warning] OpenAIEmbedder fallback to MockEmbedder: {e}")
            return _mock_embed
    return _mock_embed


def main():
    print("=" * 80)
    print(f"BENCHMARK RETRIEVAL — LAB 7: EMBEDDING & VECTOR STORE")
    print(f"Chiến lược chunking đang dùng: {CHUNKER_NAME}")
    print(f"Corpus: {DATA_DIR}")
    print("=" * 80)

    md_files = sorted(DATA_DIR.glob("*.md"))
    if not md_files:
        print(f"Lỗi: Không tìm thấy file markdown trong {DATA_DIR}")
        return 1

    # 1. Đọc file và chunking
    documents: list[Document] = []
    total_chars = 0
    doc_stats = {}

    for path in md_files:
        fm, body = parse_frontmatter(path)
        total_chars += len(body)
        chunks = CHUNKER.chunk(body)
        doc_stats[path.stem] = len(chunks)

        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{path.stem}#{i}"
            # Trải frontmatter metadata vào mọi chunk và gán doc_id
            chunk_metadata = {
                **fm,
                "doc_id": path.stem,
                "chunk_index": i,
                "total_chunks_in_doc": len(chunks),
            }
            documents.append(
                Document(
                    id=chunk_id,
                    content=chunk_text,
                    metadata=chunk_metadata,
                )
            )

    print(f"Đã đọc: {len(md_files)} tài liệu ({total_chars:,} ký tự phần thân)")
    print(f"Tổng số chunk được tạo: {len(documents)}")
    for doc_id, count in doc_stats.items():
        print(f"  - {doc_id}: {count} chunks")
    print("-" * 80)

    # 2. Nạp vào EmbeddingStore
    embed_fn = get_embedder()
    store = EmbeddingStore(embedding_fn=embed_fn)
    store.add_documents(documents)
    print(f"Đã nạp {store.get_collection_size()} chunks vào EmbeddingStore.")
    print("-" * 80)

    # Tác tử RAG để sinh câu trả lời đối chiếu
    def mock_llm_fn(prompt: str) -> str:
        # Simple RAG mock generator highlighting citations found in prompt
        citations = re.findall(r"\[(\d+)\]\s*\(([^)]+)\)", prompt)
        if citations:
            cite_str = ", ".join(f"[{c[0]}] {c[1]}" for c in citations)
            return f"According to retrieved policy documents ({cite_str}), the requested conditions and procedures apply."
        return "I could not find sufficient information in the provided context."

    agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm_fn)

    # 3. Chạy 5 Benchmark Queries
    benchmark_lines = []
    benchmark_lines.append(f"BENCHMARK RESULTS — {CHUNKER_NAME}\n")
    benchmark_lines.append(f"Total Chunks: {len(documents)}\n")

    for q in BENCHMARK_QUERIES:
        qid = q["id"]
        qtype = q["type"]
        query_text = q["query"]
        gold_ans = q["gold_answer"]
        gold_doc = q["gold_doc_id"]
        m_filter = q["filter"]
        substr = q["expected_substring"]

        print(f"\n[CÂU HỎI {qid}] ({qtype})")
        print(f"Query: {query_text}")
        print(f"Metadata Filter: {m_filter}")
        print(f"Gold Doc: {gold_doc}")
        print(f"Gold Answer: {gold_ans}")

        results = store.search_with_filter(query_text, top_k=3, metadata_filter=m_filter)

        print("\n--- Top-3 Chunks Retrieval ---")
        has_gold_doc = False
        has_substr = False

        for rank, res in enumerate(results, 1):
            r_doc_id = res.get("metadata", {}).get("doc_id", "N/A")
            r_chunk_id = res.get("metadata", {}).get("id", res.get("id", "N/A"))
            r_score = res.get("score", 0.0)
            r_content = res.get("content", "").strip()
            r_preview = r_content.replace("\n", " ")[:150]

            if r_doc_id == gold_doc:
                has_gold_doc = True
            if substr.lower() in r_content.lower():
                has_substr = True

            print(f"  Top-{rank} [Score: {r_score:.4f}] Doc: {r_doc_id} (Chunk: {r_chunk_id})")
            print(f"         Preview: {r_preview}...")

        # Sinh câu trả lời qua Agent
        agent_ans = agent.answer(query_text, top_k=3, metadata_filter=m_filter)
        agent_preview = agent_ans.replace("\n", " ")[:200]
        print(f"Agent Answer: {agent_preview}...")

        status = "RELEVANT" if (has_gold_doc and has_substr) else ("PARTIAL" if has_gold_doc else "MISSING")
        print(f"Đánh giá: [{status}] (Gold doc: {has_gold_doc}, Chứa đáp án: {has_substr})")

        benchmark_lines.append(f"\n=== Q{qid}: {query_text} ===")
        benchmark_lines.append(f"Filter: {m_filter}")
        benchmark_lines.append(f"Gold Doc: {gold_doc}")
        benchmark_lines.append(f"Status: {status}")
        for rank, res in enumerate(results, 1):
            benchmark_lines.append(f"  Top-{rank} [Score: {res.get('score', 0.0):.4f}] {res.get('metadata', {}).get('doc_id')} (id={res.get('id')}): {res.get('content', '')[:120].replace(chr(10), ' ')}...")

    # 4. A/B Test so sánh câu 2: Có filter vs Không filter
    print("\n" + "=" * 80)
    print("A/B TEST CHO CÂU 2: CÓ FILTER VS KHÔNG FILTER")
    print("=" * 80)
    q2_text = BENCHMARK_QUERIES[1]["query"]
    res_no_filter = store.search_with_filter(q2_text, top_k=3, metadata_filter=None)
    res_with_filter = store.search_with_filter(q2_text, top_k=3, metadata_filter={"audience": "buyer"})

    print("[Không lọc metadata - filter=None]:")
    for rank, r in enumerate(res_no_filter, 1):
        print(f"  Top-{rank}: Doc={r.get('metadata', {}).get('doc_id')} (Audience={r.get('metadata', {}).get('audience')}) Score={r.get('score', 0.0):.4f}")

    print("\n[Có lọc metadata - filter={'audience': 'buyer'}]:")
    for rank, r in enumerate(res_with_filter, 1):
        print(f"  Top-{rank}: Doc={r.get('metadata', {}).get('doc_id')} (Audience={r.get('metadata', {}).get('audience')}) Score={r.get('score', 0.0):.4f}")

    # Ghi file kết quả benchmark (kèm A/B test)
    benchmark_lines.append("\n" + "=" * 60)
    benchmark_lines.append("A/B TEST FOR Q2 (METADATA FILTERING)")
    benchmark_lines.append("=" * 60)
    benchmark_lines.append("[filter=None (No filter)]:")
    for rank, r in enumerate(res_no_filter, 1):
        benchmark_lines.append(f"  Top-{rank}: Doc={r.get('metadata', {}).get('doc_id')} (Audience={r.get('metadata', {}).get('audience')}) Score={r.get('score', 0.0):.4f}")
    benchmark_lines.append("\n[filter={'audience': 'buyer'} (With filter)]:")
    for rank, r in enumerate(res_with_filter, 1):
        benchmark_lines.append(f"  Top-{rank}: Doc={r.get('metadata', {}).get('doc_id')} (Audience={r.get('metadata', {}).get('audience')}) Score={r.get('score', 0.0):.4f}")

    out_file = Path("ket_qua_benchmark.txt")
    out_file.write_text("\n".join(benchmark_lines), encoding="utf-8")
    print(f"\nĐã lưu kết quả chi tiết vào: {out_file.resolve()}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
