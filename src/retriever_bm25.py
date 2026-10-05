"""
BM25-based retriever for RAG system.

This module loads document chunks and provides BM25 search functionality.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """BM25-based document retriever."""

    def __init__(self, chunks_path: str):
        """
        Initialize the retriever with document chunks.

        Args:
            chunks_path: Path to the chunks.jsonl file
        """
        self.chunks_path = Path(chunks_path)
        self.chunks = []
        self.bm25 = None
        self._load_chunks()
        self._build_index()

    def _load_chunks(self):
        """Load chunks from JSONL file."""
        print(f"Loading chunks from {self.chunks_path}...")
        with open(self.chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                chunk = json.loads(line.strip())
                self.chunks.append(chunk)
        print(f"Loaded {len(self.chunks)} chunks")

    def _build_index(self):
        """Build BM25 index from chunks."""
        print("Building BM25 index...")
        # Tokenize all chunks (simple whitespace tokenization)
        tokenized_corpus = [self._tokenize(chunk['text']) for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 index built successfully")

    def _tokenize(self, text: str) -> List[str]:
        """
        Improved tokenization that handles:
        - Course codes (CIE337 → cie + 337; CIE 337 → cie + 337)
        - Numbers
        - General text
        - Removes common stop words to reduce noise

        This is a GENERAL approach that works for any document type.

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        import re

        # Common English stop words that add noise
        stop_words = {
            'the', 'are', 'what', 'for', 'and', 'this', 'that', 'with', 'from',
            'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can',
            'does', 'did', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again', 'further',
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'only', 'own', 'same', 'than', 'too', 'very'
        }

        # Lowercase
        text = text.lower()

        # IMPORTANT: Normalize patterns like "CIE337" → "CIE 337" (split letters from numbers)
        # This handles course codes in various formats generically
        text = re.sub(r'([a-z]+)(\d+)', r'\1 \2', text)

        # Split on whitespace and punctuation, but keep alphanumeric together
        tokens = re.findall(r'\b\w+\b', text)

        # Filter: remove stop words, keep tokens > 2 chars or numbers
        tokens = [
            t for t in tokens
            if (t not in stop_words) and (len(t) > 2 or t.isdigit())
        ]

        return tokens

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant chunks for a query.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of chunk dictionaries with added 'score' field
        """
        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Prepare results
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['score'] = float(scores[idx])
            results.append(chunk)

        return results

    def retrieve_with_scores(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Retrieve top-k chunks with their scores.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of (chunk, score) tuples
        """
        results = self.retrieve(query, top_k)
        return [(chunk, chunk['score']) for chunk in results]


def main():
    """Example usage of the retriever."""
    # Initialize retriever
    retriever = BM25Retriever("data/processed/engineering/chunks.jsonl")

    # Example queries
    queries = [
        "What are the prerequisites for CIE337?",
        "How many credits is NANENG301 worth?",
        "What does the attendance policy state?",
    ]

    print("\n" + "="*80)
    print("TESTING RETRIEVER")
    print("="*80)

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 80)

        results = retriever.retrieve(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"\n[Rank {i}] Score: {result['score']:.4f}")
            print(f"ID: {result['id']}")
            print(f"Doc Type: {result['doc_type']}, Page: {result['page_start']}-{result['page_end']}")
            print(f"Text preview: {result['text'][:200]}...")

        print("\n" + "="*80)


if __name__ == "__main__":
    main()
