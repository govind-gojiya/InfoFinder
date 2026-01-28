"""Hybrid search service combining vector and keyword search with RRF."""

from typing import Optional
from collections import defaultdict

from models.schemas import Document, SearchResult
from services.vector_store import VectorStoreService
from services.keyword_search import KeywordSearchService
from services.embeddings import EmbeddingService
import config


class HybridSearchService:
    """
    Hybrid search combining vector similarity and BM25 keyword search.
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        chat_id: str,
        embedding_service: EmbeddingService = None,
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5,
        rrf_k: int = None
    ):
        """
        Initialize the hybrid search service.
        
        Args:
            chat_id: Unique identifier for the chat
            embedding_service: Optional embedding service instance
            vector_weight: Weight for vector search results (0-1)
            keyword_weight: Weight for keyword search results (0-1)
            rrf_k: RRF constant (higher values give less weight to high ranks)
        """
        self.chat_id = chat_id
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Normalize weights
        total_weight = vector_weight + keyword_weight
        self.vector_weight = vector_weight / total_weight
        self.keyword_weight = keyword_weight / total_weight
        
        self.rrf_k = rrf_k or config.RRF_K
        
        # Initialize search services
        self.vector_store = VectorStoreService(chat_id, self.embedding_service)
        self.keyword_search = KeywordSearchService(chat_id)
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to both vector and keyword indexes.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Add to both indexes
        vector_ids = self.vector_store.add_documents(documents)
        keyword_ids = self.keyword_search.add_documents(documents)
        
        return vector_ids
    
    def search(
        self,
        query: str,
        top_k: int = None,
        use_vector: bool = True,
        use_keyword: bool = True
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector and keyword search.
        
        Args:
            query: Query text
            top_k: Number of results to return
            use_vector: Whether to use vector search
            use_keyword: Whether to use keyword search
            
        Returns:
            List of SearchResult objects sorted by combined score
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Collect results from both search methods
        vector_results = []
        keyword_results = []
        
        if use_vector:
            vector_results = self.vector_store.search(query, top_k=top_k * 2)
        
        if use_keyword:
            keyword_results = self.keyword_search.search(query, top_k=top_k * 2)
        
        # If only one search type is used, return its results directly
        if not use_vector:
            return keyword_results[:top_k]
        if not use_keyword:
            return vector_results[:top_k]
        
        # Combine using RRF
        combined_results = self._rrf_fusion(
            vector_results,
            keyword_results,
            top_k
        )
        
        return combined_results
    
    def _rrf_fusion(
        self,
        vector_results: list[SearchResult],
        keyword_results: list[SearchResult],
        top_k: int
    ) -> list[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum(1 / (k + rank)) for each result list
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            top_k: Number of results to return
            
        Returns:
            Combined and sorted list of SearchResult objects
        """
        # Calculate RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.document.id
            rrf_score = self.vector_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[doc_id] += rrf_score
            doc_map[doc_id] = result.document
        
        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result.document.id
            rrf_score = self.keyword_weight * (1.0 / (self.rrf_k + rank))
            rrf_scores[doc_id] += rrf_score
            doc_map[doc_id] = result.document
        
        # Sort by RRF score
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:top_k]
        
        # Create final results
        results = []
        for doc_id in sorted_doc_ids:
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=rrf_scores[doc_id],
                search_type="hybrid"
            ))
        
        return results
    
    def search_with_scores(
        self,
        query: str,
        top_k: int = None
    ) -> tuple[list[SearchResult], dict]:
        """
        Perform hybrid search and return detailed scoring information.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            Tuple of (results, score_details)
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Get results from both methods
        vector_results = self.vector_store.search(query, top_k=top_k * 2)
        keyword_results = self.keyword_search.search(query, top_k=top_k * 2)
        
        # Create score details
        score_details = {
            "vector_results": [
                {"doc_id": r.document.id, "score": r.score}
                for r in vector_results
            ],
            "keyword_results": [
                {"doc_id": r.document.id, "score": r.score}
                for r in keyword_results
            ]
        }
        
        # Combine using RRF
        combined_results = self._rrf_fusion(vector_results, keyword_results, top_k)
        
        score_details["hybrid_results"] = [
            {"doc_id": r.document.id, "score": r.score}
            for r in combined_results
        ]
        
        return combined_results, score_details
    
    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = None
    ) -> list[SearchResult]:
        """
        Perform multi-query search with RRF fusion across all query results.
        
        For each query:
        1. Perform vector search
        2. Perform keyword search
        3. Apply RRF to combine vector + keyword results
        
        Then apply RRF again across all query results.
        
        Args:
            queries: List of queries (original + similar queries)
            top_k: Number of final results to return
            
        Returns:
            List of SearchResult objects after multi-query RRF fusion
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        if not queries:
            return []
        
        # If only one query, use regular search
        if len(queries) == 1:
            return self.search(queries[0], top_k=top_k)
        
        # Collect RRF results for each query
        all_query_results: list[list[SearchResult]] = []
        
        for query in queries:
            # Get hybrid search results for this query (already RRF fused)
            query_results = self.search(query, top_k=top_k * 2)
            if query_results:
                all_query_results.append(query_results)
        
        if not all_query_results:
            return []
        
        # Apply RRF fusion across all query results
        final_results = self._multi_query_rrf_fusion(all_query_results, top_k)
        
        return final_results
    
    def _multi_query_rrf_fusion(
        self,
        query_results_list: list[list[SearchResult]],
        top_k: int
    ) -> list[SearchResult]:
        """
        Apply RRF fusion across multiple query results.
        
        Each query's results are treated as a separate ranking list.
        
        Args:
            query_results_list: List of results from each query
            top_k: Number of results to return
            
        Returns:
            Combined and sorted list of SearchResult objects
        """
        # Calculate RRF scores across all query results
        rrf_scores: dict[str, float] = defaultdict(float)
        doc_map: dict[str, Document] = {}
        
        # Weight for each query (equal weight for all queries)
        num_queries = len(query_results_list)
        query_weight = 1.0 / num_queries
        
        # Process results from each query
        for query_results in query_results_list:
            for rank, result in enumerate(query_results, start=1):
                doc_id = result.document.id
                # RRF score contribution from this query
                rrf_score = query_weight * (1.0 / (self.rrf_k + rank))
                rrf_scores[doc_id] += rrf_score
                doc_map[doc_id] = result.document
        
        # Sort by combined RRF score
        sorted_doc_ids = sorted(
            rrf_scores.keys(),
            key=lambda x: rrf_scores[x],
            reverse=True
        )[:top_k]
        
        # Create final results
        results = []
        for doc_id in sorted_doc_ids:
            results.append(SearchResult(
                document=doc_map[doc_id],
                score=rrf_scores[doc_id],
                search_type="multi_query_hybrid"
            ))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self.vector_store.get_document(doc_id)
    
    def get_all_documents(self) -> list[Document]:
        """Get all documents."""
        return self.vector_store.get_all_documents()
    
    def document_count(self) -> int:
        """Get the number of documents."""
        return self.vector_store.document_count()
    
    def delete_documents(self, doc_ids: list[str]):
        """Delete documents from both indexes."""
        self.vector_store.delete_documents(doc_ids)
        self.keyword_search.delete_documents(doc_ids)
    
    def clear(self):
        """Clear all documents from both indexes."""
        self.vector_store.clear()
        self.keyword_search.clear()

