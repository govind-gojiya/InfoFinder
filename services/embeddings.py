"""Embedding service for generating vector embeddings."""

from typing import Union
import numpy as np
from sentence_transformers import SentenceTransformer

from models.schemas import Document
import config


class EmbeddingService:
    """Service for generating embeddings using sentence-transformers."""
    
    _instance = None
    _model = None
    
    def __new__(cls, model_name: str = None):
        """Singleton pattern to reuse the model across instances."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_name: str = None):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name or config.EMBEDDING_MODEL
        
        if EmbeddingService._model is None:
            EmbeddingService._model = SentenceTransformer(self.model_name)
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the embedding model."""
        return EmbeddingService._model
    
    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Numpy array of the embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings (n_texts x dimension)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )
    
    def embed_document(self, document: Document) -> np.ndarray:
        """
        Generate embedding for a document.
        
        Args:
            document: Document to embed
            
        Returns:
            Numpy array of the embedding
        """
        return self.embed_text(document.content)
    
    def embed_documents(self, documents: list[Document], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for multiple documents.
        
        Args:
            documents: List of documents to embed
            batch_size: Batch size for encoding
            
        Returns:
            Numpy array of embeddings (n_docs x dimension)
        """
        texts = [doc.content for doc in documents]
        return self.embed_texts(texts, batch_size)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        Note: Some models have different encoding for queries vs documents.
        
        Args:
            query: Query text to embed
            
        Returns:
            Numpy array of the embedding
        """
        return self.embed_text(query)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

