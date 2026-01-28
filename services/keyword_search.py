"""Keyword search service using BM25."""

import re
import json
from pathlib import Path
from typing import Optional
from rank_bm25 import BM25Okapi

from models.schemas import Document, SearchResult, ContentType
import config


class KeywordSearchService:
    """Service for keyword-based search using BM25."""
    
    def __init__(self, chat_id: str):
        """
        Initialize the keyword search service for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat
        """
        self.chat_id = chat_id
        self.store_path = config.VECTOR_STORES_DIR / chat_id
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        self.documents: list[Document] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        
        self._load_index()
    
    def _tokenize(self, text: str) -> list[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and split
        tokens = re.findall(r'\b\w+\b', text)
        
        # Remove very short tokens and stopwords
        stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for',
            'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
            'that', 'the', 'to', 'was', 'were', 'will', 'with'
        }
        tokens = [t for t in tokens if len(t) > 1 and t not in stopwords]
        
        return tokens
    
    def _load_index(self):
        """Load the BM25 index from disk."""
        index_path = self.store_path / "bm25_index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                    
                    # Restore documents
                    for doc_data in data.get("documents", []):
                        doc = Document(
                            id=doc_data["id"],
                            content=doc_data["content"],
                            content_type=ContentType(doc_data["content_type"]),
                            metadata=doc_data.get("metadata", {}),
                            source_file=doc_data.get("source_file", ""),
                            page_number=doc_data.get("page_number", 0),
                            chunk_index=doc_data.get("chunk_index", 0)
                        )
                        self.documents.append(doc)
                    
                    # Restore tokenized corpus
                    self.tokenized_corpus = data.get("tokenized_corpus", [])
                    
                    # Rebuild BM25 index
                    if self.tokenized_corpus:
                        self.bm25 = BM25Okapi(self.tokenized_corpus)
            except Exception:
                self.documents = []
                self.tokenized_corpus = []
                self.bm25 = None
    
    def _save_index(self):
        """Save the BM25 index to disk."""
        index_path = self.store_path / "bm25_index.json"
        
        data = {
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "content_type": doc.content_type.value,
                    "metadata": doc.metadata,
                    "source_file": doc.source_file,
                    "page_number": doc.page_number,
                    "chunk_index": doc.chunk_index
                }
                for doc in self.documents
            ],
            "tokenized_corpus": self.tokenized_corpus
        }
        
        with open(index_path, "w") as f:
            json.dump(data, f)
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the BM25 index.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        ids = []
        for doc in documents:
            self.documents.append(doc)
            tokens = self._tokenize(doc.content)
            self.tokenized_corpus.append(tokens)
            ids.append(doc.id)
        
        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        self._save_index()
        
        return ids
    
    def search(self, query: str, top_k: int = None) -> list[SearchResult]:
        """
        Search for documents using BM25.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        if not self.bm25 or not self.documents:
            return []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k results
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Normalize scores to 0-1 range
        max_score = max(scores) if max(scores) > 0 else 1
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                results.append(SearchResult(
                    document=self.documents[idx],
                    score=scores[idx] / max_score,  # Normalize
                    search_type="keyword"
                ))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None
    
    def get_all_documents(self) -> list[Document]:
        """Get all documents."""
        return self.documents
    
    def document_count(self) -> int:
        """Get the number of documents."""
        return len(self.documents)
    
    def delete_documents(self, doc_ids: list[str]):
        """Delete documents by IDs."""
        if not doc_ids:
            return
        
        doc_ids_set = set(doc_ids)
        
        # Filter out deleted documents
        new_documents = []
        new_tokenized = []
        
        for i, doc in enumerate(self.documents):
            if doc.id not in doc_ids_set:
                new_documents.append(doc)
                new_tokenized.append(self.tokenized_corpus[i])
        
        self.documents = new_documents
        self.tokenized_corpus = new_tokenized
        
        # Rebuild BM25 index
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None
        
        self._save_index()
    
    def clear(self):
        """Clear all documents."""
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._save_index()

