"""Vector store service using ChromaDB."""

import json
from pathlib import Path
from typing import Optional
import chromadb
from chromadb.config import Settings
import numpy as np

from models.schemas import Document, SearchResult, ContentType
from services.embeddings import EmbeddingService
import config


class VectorStoreService:
    """Service for managing vector stores using ChromaDB."""
    
    def __init__(self, chat_id: str, embedding_service: EmbeddingService = None):
        """
        Initialize the vector store for a specific chat.
        
        Args:
            chat_id: Unique identifier for the chat
            embedding_service: Optional embedding service instance
        """
        self.chat_id = chat_id
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Create persistent storage path for this chat
        self.store_path = config.VECTOR_STORES_DIR / chat_id
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.store_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection for this chat
        self.collection = self.client.get_or_create_collection(
            name=f"chat_{chat_id}",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Local document cache for retrieval
        self._document_cache: dict[str, Document] = {}
        self._load_document_cache()
    
    def _load_document_cache(self):
        """Load document cache from disk."""
        cache_path = self.store_path / "document_cache.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    cache_data = json.load(f)
                    for doc_id, doc_data in cache_data.items():
                        self._document_cache[doc_id] = Document(
                            id=doc_data["id"],
                            content=doc_data["content"],
                            content_type=ContentType(doc_data["content_type"]),
                            metadata=doc_data.get("metadata", {}),
                            source_file=doc_data.get("source_file", ""),
                            page_number=doc_data.get("page_number", 0),
                            chunk_index=doc_data.get("chunk_index", 0)
                        )
            except Exception:
                self._document_cache = {}
    
    def _save_document_cache(self):
        """Save document cache to disk."""
        cache_path = self.store_path / "document_cache.json"
        cache_data = {}
        for doc_id, doc in self._document_cache.items():
            cache_data[doc_id] = {
                "id": doc.id,
                "content": doc.content,
                "content_type": doc.content_type.value,
                "metadata": doc.metadata,
                "source_file": doc.source_file,
                "page_number": doc.page_number,
                "chunk_index": doc.chunk_index
            }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)
    
    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents to add
            
        Returns:
            List of document IDs
        """
        if not documents:
            return []
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_documents(documents)
        
        # Prepare data for ChromaDB
        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [
            {
                "source_file": doc.source_file,
                "page_number": doc.page_number,
                "chunk_index": doc.chunk_index,
                "content_type": doc.content_type.value
            }
            for doc in documents
        ]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        # Update document cache
        for doc in documents:
            self._document_cache[doc.id] = doc
        
        self._save_document_cache()
        
        return ids
    
    def search(
        self,
        query: str,
        top_k: int = None,
        filter_metadata: dict = None
    ) -> list[SearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query: Query text
            top_k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of SearchResult objects
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query)
        
        # Build where clause if filter provided
        where = None
        if filter_metadata:
            where = filter_metadata
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()) if self.collection.count() > 0 else top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["ids"] or not results["ids"][0]:
            return []
        
        # Convert to SearchResult objects
        search_results = []
        for idx, doc_id in enumerate(results["ids"][0]):
            # Get document from cache or create from results
            if doc_id in self._document_cache:
                document = self._document_cache[doc_id]
            else:
                metadata = results["metadatas"][0][idx] if results["metadatas"] else {}
                document = Document(
                    id=doc_id,
                    content=results["documents"][0][idx] if results["documents"] else "",
                    content_type=ContentType(metadata.get("content_type", "text")),
                    metadata=metadata,
                    source_file=metadata.get("source_file", ""),
                    page_number=metadata.get("page_number", 0),
                    chunk_index=metadata.get("chunk_index", 0)
                )
            
            # Convert distance to similarity score (ChromaDB returns L2 distance for cosine)
            distance = results["distances"][0][idx] if results["distances"] else 0
            # For cosine distance in ChromaDB: similarity = 1 - distance
            similarity = 1 - distance
            
            search_results.append(SearchResult(
                document=document,
                score=similarity,
                search_type="vector"
            ))
        
        return search_results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a document by ID."""
        return self._document_cache.get(doc_id)
    
    def get_all_documents(self) -> list[Document]:
        """Get all documents in the store."""
        return list(self._document_cache.values())
    
    def document_count(self) -> int:
        """Get the number of documents in the store."""
        return self.collection.count()
    
    def delete_documents(self, doc_ids: list[str]):
        """Delete documents by IDs."""
        if not doc_ids:
            return
        
        self.collection.delete(ids=doc_ids)
        
        for doc_id in doc_ids:
            self._document_cache.pop(doc_id, None)
        
        self._save_document_cache()
    
    def clear(self):
        """Clear all documents from the store."""
        # Delete and recreate collection
        self.client.delete_collection(f"chat_{self.chat_id}")
        self.collection = self.client.get_or_create_collection(
            name=f"chat_{self.chat_id}",
            metadata={"hnsw:space": "cosine"}
        )
        self._document_cache.clear()
        self._save_document_cache()
    
    @staticmethod
    def delete_store(chat_id: str):
        """Delete the entire vector store for a chat."""
        import shutil
        store_path = config.VECTOR_STORES_DIR / chat_id
        if store_path.exists():
            shutil.rmtree(store_path)

