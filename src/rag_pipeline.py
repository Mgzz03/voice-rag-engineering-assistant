"""
RAG (Retrieval-Augmented Generation) Pipeline.

Combines BM25 retrieval with LLM generation to answer questions.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retriever_bm25 import BM25Retriever
from src.llm_client import LLMClient


class RAGPipeline:
    """RAG pipeline combining retrieval and generation."""

    def __init__(
        self,
        chunks_path: str,
        llm_base_url: str = "http://172.31.16.1:1234",
        max_chunks: int = 10,
        min_chunks: int = 1,
        score_threshold: float = 0.5,
        score_gap_threshold: float = 3.0,
        min_score_for_llm: float = 2.0
    ):
        """
        Initialize RAG pipeline with adaptive chunk selection.

        Args:
            chunks_path: Path to chunks.jsonl file
            llm_base_url: URL where LLM is running
            max_chunks: Maximum chunks to send (default 10)
            min_chunks: Minimum chunks to send if any found (default 1)
            score_threshold: Minimum BM25 score to include a chunk (default 0.5)
            score_gap_threshold: Max score drop before stopping (default 3.0)
            min_score_for_llm: Minimum score of best chunk to proceed with LLM (default 2.0)
        """
        self.retriever = BM25Retriever(chunks_path)
        self.llm = LLMClient(llm_base_url)
        self.max_chunks = max_chunks
        self.min_chunks = min_chunks
        self.score_threshold = score_threshold
        self.score_gap_threshold = score_gap_threshold
        self.min_score_for_llm = min_score_for_llm

        # Store conversation history as actual messages
        self.conversation_history = []  # List of {"role": "user"/"assistant", "content": "..."}

        # System prompt for the LLM
        self.system_prompt = """You are a helpful assistant answering questions about university documents.

Instructions:
- Answer ONLY based on the provided context
- Be concise and direct
- If the context doesn't contain the answer, say "I don't have enough information to answer that question."
- Do not make up information"""

    def query(
        self,
        question: str,
        return_sources: bool = False,
        use_history: bool = True,
        max_tokens: int = 256,
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG with adaptive chunk selection.

        Args:
            question: User's question
            return_sources: Whether to return retrieved sources
            use_history: Whether to use conversation history for context
            max_tokens: Max tokens for LLM response
            temperature: LLM temperature (lower = more focused)

        Returns:
            Dictionary with 'answer', 'num_chunks_used', and optionally 'sources'
        """
        # Step 1: Retrieve candidate chunks
        candidate_chunks = self.retriever.retrieve(question, top_k=self.max_chunks * 2)

        # Step 2: Adaptively select chunks based on scores
        retrieved_chunks = self._select_chunks_adaptively(candidate_chunks)

        # Step 3: Check if we have good enough chunks to proceed
        if not retrieved_chunks or retrieved_chunks[0]['score'] < self.min_score_for_llm:
            # Not enough relevant information found
            return {
                "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                "num_chunks_used": 0,
                "sources": [] if return_sources else None
            }

        # Step 4: Format context from retrieved chunks
        context = self._format_context(retrieved_chunks)

        # Step 5: Create prompt with context
        prompt = self._create_prompt(question, context)

        # Step 6: Generate answer, passing conversation history to LLM
        answer = self.llm.generate(
            prompt=prompt,
            system_prompt=self.system_prompt,
            conversation_history=self.conversation_history if use_history else None,
            max_tokens=max_tokens,
            temperature=temperature
        )

        # Step 7: Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": answer})

            # Keep only last 6 messages (3 Q&A pairs)
            if len(self.conversation_history) > 6:
                self.conversation_history = self.conversation_history[-6:]

        # Prepare response
        result = {
            "answer": answer,
            "num_chunks_used": len(retrieved_chunks)
        }

        if return_sources:
            result["sources"] = [
                {
                    "doc_type": chunk["doc_type"],
                    "page": f"{chunk['page_start']}-{chunk['page_end']}",
                    "score": chunk["score"],
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"]
                }
                for chunk in retrieved_chunks
            ]

        return result

    def _select_chunks_adaptively(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Adaptively select chunks based on scores.

        Strategy:
        1. Keep chunks above score_threshold
        2. Stop if score gap is too large (indicates quality drop)
        3. Respect min_chunks and max_chunks bounds

        Args:
            chunks: Candidate chunks sorted by score (descending)

        Returns:
            Selected chunks to use for context
        """
        if not chunks:
            return []

        selected = []
        prev_score = None

        for chunk in chunks:
            score = chunk['score']

            # Always include first chunk if it meets threshold
            if len(selected) == 0:
                if score >= self.score_threshold:
                    selected.append(chunk)
                    prev_score = score
                else:
                    break  # First chunk too low, skip all
                continue

            # Check if we've hit max
            if len(selected) >= self.max_chunks:
                break

            # Check score threshold
            if score < self.score_threshold:
                break

            # Check for large score drop (quality cliff)
            if prev_score is not None and (prev_score - score) > self.score_gap_threshold:
                break

            selected.append(chunk)
            prev_score = score

        # Ensure we have at least min_chunks if available
        while len(selected) < self.min_chunks and len(selected) < len(chunks):
            next_chunk = chunks[len(selected)]
            selected.append(next_chunk)

        return selected

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def _format_context(self, chunks: List[Dict[str, Any]], max_chunk_chars: int = 600) -> str:
        """
        Format retrieved chunks into context string.

        Args:
            chunks: Retrieved chunks
            max_chunk_chars: Maximum characters per chunk (to keep context manageable)

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant information found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_type = chunk["doc_type"]
            page = f"{chunk['page_start']}-{chunk['page_end']}"
            text = chunk["text"]

            # Truncate if too long
            if len(text) > max_chunk_chars:
                text = text[:max_chunk_chars] + "..."

            context_parts.append(
                f"[Source {i} - {doc_type}, page {page}]\n{text}"
            )

        return "\n\n".join(context_parts)

    def _create_prompt(self, question: str, context: str) -> str:
        """
        Create the prompt for the LLM.

        Args:
            question: User's question
            context: Retrieved context

        Returns:
            Formatted prompt
        """
        return f"""Context:
{context}

Question: {question}

Answer based on the context above:"""


def main():
    """Test the RAG pipeline with adaptive chunk selection."""
    print("Initializing RAG Pipeline with Adaptive Selection...")
    print("="*80)

    # Initialize pipeline
    rag = RAGPipeline(
        chunks_path="data/processed/engineering/chunks.jsonl",
        max_chunks=10,
        min_chunks=1,
        score_threshold=0.5,
        score_gap_threshold=3.0,
        min_score_for_llm=2.0
    )

    print("✓ RAG Pipeline initialized")
    print(f"  Adaptive selection: 1-10 chunks")
    print(f"  Early exit: min score {2.0}")
    print(f"  LLM handles conversation context\n")

    # Test questions
    test_questions = [
        "What are the prerequisites for CIE337?",
        "How many credits is NANENG301 worth?",
        "What does the attendance policy state?",
    ]

    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {i}: {question}")
        print("="*80)

        # Get answer with sources
        result = rag.query(question, return_sources=True)

        print(f"\nAnswer:\n{result['answer']}")

        num_chunks = result['num_chunks_used']
        print(f"\n✓ Adaptively selected {num_chunks}/{len(result['sources'])} chunks")
        print(f"\nSources used:")
        for j, source in enumerate(result['sources'], 1):
            print(f"  {j}. {source['doc_type']} (page {source['page']}, score={source['score']:.2f})")

    print("\n" + "="*80)
    print("RAG Pipeline test complete!")


if __name__ == "__main__":
    main()
