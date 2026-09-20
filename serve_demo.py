from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# UTF-8 stdout
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PORT = 8000
DIRECTORY = Path(__file__).resolve().parent

# Ensure project root is in sys.path
if str(DIRECTORY) not in sys.path:
    sys.path.insert(0, str(DIRECTORY))

# Load environment variables from .env
def load_env() -> None:
    env_file = DIRECTORY / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

load_env()

from src.chunking import (
    FixedSizeChunker,
    MarkdownHeadingChunker,
    RecursiveChunker,
    SentenceChunker,
)
from src.embeddings import _mock_embed
from src.store import Document, EmbeddingStore

# Pre-cache stores for fast interactive querying
STORES: dict[str, EmbeddingStore] = {}

def extract_frontmatter(text: str) -> tuple[dict[str, str], str]:
    parts = text.split("---", 2)
    meta = {}
    content = text
    if len(parts) >= 3:
        fm = parts[1]
        content = parts[2].strip()
        for k, v in re.findall(r"^(\w+):\s*(.+)$", fm, re.M):
            meta[k.strip()] = v.strip()
    return meta, content

def get_or_build_store(chunker_name: str) -> EmbeddingStore:
    if chunker_name in STORES:
        return STORES[chunker_name]

    data_dir = DIRECTORY / "data" / "ebay-policies"
    chunker_map = {
        "MarkdownHeadingChunker": MarkdownHeadingChunker(),
        "RecursiveChunker": RecursiveChunker(chunk_size=400),
        "SentenceChunker": SentenceChunker(max_sentences_per_chunk=3),
        "FixedSizeChunker": FixedSizeChunker(chunk_size=400, overlap=50),
    }
    chunker = chunker_map.get(chunker_name, chunker_map["MarkdownHeadingChunker"])
    store = EmbeddingStore(embedding_fn=_mock_embed)

    docs: list[Document] = []
    for md_file in sorted(data_dir.glob("*.md")):
        meta, body = extract_frontmatter(md_file.read_text(encoding="utf-8"))
        chunks = chunker.chunk(body)
        for i, ch in enumerate(chunks):
            doc_id = f"{md_file.stem}#{i}"
            doc_meta = dict(meta)
            doc_meta["doc_id"] = md_file.stem
            docs.append(Document(id=doc_id, content=ch, metadata=doc_meta))

    store.add_documents(docs)
    STORES[chunker_name] = store
    return store


def call_llm(prompt: str, context: str) -> tuple[str, str]:
    """Call LLM via OpenRouter, Gemini, or OpenAI API, with extractive fallback."""
    # Check for OpenRouter API key
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        for k, v in os.environ.items():
            if v.startswith("sk-or-v1-"):
                openrouter_key = v
                break

    if openrouter_key:
        model = os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free")
        # Try primary model and backup free models
        models_to_try = [model, "nex-agi/nex-n2.5-pro:free", "qwen/qwen3.8-27b:free", "google/gemma-4-31b-it:free"]
        for m in models_to_try:
            try:
                payload = {
                    "model": m,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Bạn là trợ lý AI chuyên gia về chính sách của eBay. "
                                "Dựa vào các đoạn trích dẫn được cung cấp, hãy trả lời câu hỏi của người dùng "
                                "bằng tiếng Việt một cách rõ ràng, súc tích, chính xác và có căn cứ điều khoản."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f"ĐOẠN TRÍCH CHÍNH SÁCH EBAY:\n{context}\n\nCÂU HỎI: {prompt}\n\nTRẢ LỜI:",
                        },
                    ],
                }
                req = urllib.request.Request(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {openrouter_key}",
                        "Content-Type": "application/json",
                    },
                    data=json.dumps(payload).encode("utf-8"),
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    answer = data["choices"][0]["message"]["content"].strip()
                    if answer:
                        return answer, f"OpenRouter ({m})"
            except Exception as e:
                print(f"[OpenRouter {m} Error]: {e}")
                continue

    # Fallback to smart grounded extractive summary
    first_chunk = context.split("\n\n")[0] if context else ""
    clean_chunk = re.sub(r"^\[.*?\]\s*", "", first_chunk).strip()
    sentences = [s.strip() for s in clean_chunk.split(".") if len(s.strip()) > 15]
    summary_body = ". ".join(sentences[:3]) + ("." if sentences else clean_chunk[:250])

    fallback_answer = (
        f"Theo điều khoản chính sách có liên quan nhất trong hệ thống dữ liệu: "
        f"<b>{summary_body}</b>"
    )
    return fallback_answer, "Grounded Policy Extractor (RAG Retrieval)"


class CustomHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DIRECTORY), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
            has_or = bool(openrouter_key or any(v.startswith("sk-or-v1-") for v in os.environ.values()))
            status_data = {
                "has_openrouter": has_or,
                "has_openai": bool(os.getenv("OPENAI_API_KEY")),
                "has_gemini": bool(os.getenv("GEMINI_API_KEY")),
                "model": os.getenv("OPENROUTER_MODEL", "nex-agi/nex-n2.5-mini:free"),
                "status": "ready",
            }
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(status_data, ensure_ascii=False).encode("utf-8"))
            return

        super().do_GET()

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
                query = data.get("query", "").strip()
                strategy = data.get("strategy", "MarkdownHeadingChunker")
                audience_filter = data.get("filter", "buyer").strip().lower()

                if not query:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b'{"error": "Query cannot be empty"}')
                    return

                store = get_or_build_store(strategy)

                filt = {"audience": audience_filter} if audience_filter in ("buyer", "seller") else None
                if filt:
                    with_filter = store.search_with_filter(query, top_k=3, metadata_filter=filt)
                else:
                    with_filter = store.search(query, top_k=3)

                no_filter = store.search(query, top_k=3)

                context_pieces = [f"[{c['id']}] {c['content']}" for c in with_filter[:2]]
                context_str = "\n\n".join(context_pieces)

                answer, model_used = call_llm(query, context_str)

                citations = [c["id"] for c in with_filter[:2]]

                resp_data = {
                    "status": "success",
                    "query": query,
                    "strategy": strategy,
                    "filter": audience_filter,
                    "answer": answer,
                    "model_used": model_used,
                    "citations": citations,
                    "with_filter": with_filter,
                    "no_filter": no_filter,
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp_data, ensure_ascii=False).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                err_resp = {"error": str(e)}
                self.wfile.write(json.dumps(err_resp, ensure_ascii=False).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()


def main():
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, CustomHandler)
    url = f"http://localhost:{PORT}/demo.html"
    print("=" * 70)
    print(f"🚀 LAB 7 LIVE RAG API & DASHBOARD SERVER")
    print(f"URL Giao diện: {url}")
    print(f"API Endpoint : http://localhost:{PORT}/api/query (Hỗ trợ truy vấn trực tiếp)")
    print(f"Nhấn Ctrl + C để dừng server.")
    print("=" * 70)

    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng demo server.")
        httpd.server_close()


if __name__ == "__main__":
    main()

