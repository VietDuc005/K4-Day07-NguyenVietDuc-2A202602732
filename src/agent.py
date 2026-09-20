from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict | None = None) -> str:
        if self.store.get_collection_size() == 0:
            return "Knowledge base is empty. No documents available to answer the question."

        if metadata_filter:
            results = self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
        else:
            results = self.store.search(question, top_k=top_k)

        if not results:
            return "No relevant information found in the knowledge base."

        # Build context with numbered citations [1], [2], ...
        context_blocks = []
        for i, res in enumerate(results, start=1):
            source = res["metadata"].get("source", res.get("id", f"doc_{i}"))
            context_blocks.append(f"[{i}] (Source: {source})\n{res['content']}")

        context_str = "\n\n".join(context_blocks)

        prompt = (
            f"You are a helpful knowledge assistant. Answer the user's question based strictly on the provided context.\n"
            f"If the answer cannot be found in the context, state clearly that you do not know.\n"
            f"Cite the relevant sources using their numbers like [1], [2] where appropriate.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )

        return self.llm_fn(prompt)
