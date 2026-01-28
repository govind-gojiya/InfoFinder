"""Reranking service using cross-encoder models."""

from typing import Optional
from sentence_transformers import CrossEncoder

from models.schemas import SearchResult
import config


class RerankerService:
    """Service for reranking search results using cross-encoder models."""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = None):
        """Singleton pattern to reuse the model across instances."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        """
        Initialize the reranker service.
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name or config.RERANKER_MODEL
        
        if RerankerService._model is None:
            RerankerService._model = CrossEncoder(self.model_name)
    
    @property
    def model(self) -> CrossEncoder:
        """Get the cross-encoder model."""
        return RerankerService._model
    
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = None
    ) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder.
        
        Args:
            query: Query text
            results: List of SearchResult objects to rerank
            top_k: Number of results to return after reranking
            
        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k or config.TOP_K_RERANK
        
        if not results:
            return []
        
        if len(results) <= 1:
            return results[:top_k]
        
        # Prepare query-document pairs
        pairs = [(query, result.document.content) for result in results]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Combine results with new scores
        scored_results = list(zip(results, scores))
        
        # Sort by cross-encoder score (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Create new SearchResult objects with updated scores
        reranked_results = []
        for result, score in scored_results[:top_k]:
            reranked_results.append(SearchResult(
                document=result.document,
                score=float(score),
                search_type="reranked"
            ))
        
        return reranked_results
    
    def rerank_with_original_scores(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = None,
        rerank_weight: float = 0.7,
        original_weight: float = 0.3
    ) -> list[SearchResult]:
        """
        Rerank results and combine with original scores.
        
        Args:
            query: Query text
            results: List of SearchResult objects to rerank
            top_k: Number of results to return
            rerank_weight: Weight for reranker score
            original_weight: Weight for original retrieval score
            
        Returns:
            Reranked list of SearchResult objects
        """
        top_k = top_k or config.TOP_K_RERANK
        
        if not results:
            return []
        
        if len(results) <= 1:
            return results[:top_k]
        
        # Prepare query-document pairs
        pairs = [(query, result.document.content) for result in results]
        
        # Get cross-encoder scores
        rerank_scores = self.model.predict(pairs)
        
        # Normalize rerank scores to 0-1 range
        min_score = min(rerank_scores)
        max_score = max(rerank_scores)
        score_range = max_score - min_score if max_score != min_score else 1
        
        normalized_rerank_scores = [
            (score - min_score) / score_range
            for score in rerank_scores
        ]
        
        # Combine scores
        combined_results = []
        for idx, result in enumerate(results):
            combined_score = (
                rerank_weight * normalized_rerank_scores[idx] +
                original_weight * result.score
            )
            combined_results.append((result, combined_score, normalized_rerank_scores[idx]))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Create final results
        reranked_results = []
        for result, combined_score, rerank_score in combined_results[:top_k]:
            reranked_results.append(SearchResult(
                document=result.document,
                score=combined_score,
                search_type="reranked"
            ))
        
        return reranked_results
    
    def get_relevance_score(self, query: str, text: str) -> float:
        """
        Get relevance score for a single query-text pair.
        
        Args:
            query: Query text
            text: Document text
            
        Returns:
            Relevance score
        """
        score = self.model.predict([(query, text)])[0]
        return float(score)

