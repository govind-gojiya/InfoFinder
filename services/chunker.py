"""Document chunking service for splitting documents into smaller pieces."""

from typing import Generator
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models.schemas import Document, ContentType
import config


class DocumentChunker:
    """Service for chunking documents into smaller pieces for embedding."""
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        """
        Initialize the document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size or config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        chunked_docs = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            chunked_docs.extend(chunks)
        
        return chunked_docs
    
    def chunk_document(self, document: Document) -> list[Document]:
        """
        Chunk a single document into smaller pieces.
        
        Args:
            document: Document object to chunk
            
        Returns:
            List of chunked Document objects
        """
        # Don't chunk images - they're already discrete units
        if document.content_type == ContentType.IMAGE:
            return [document]
        
        # For tables, only chunk if they're very large
        if document.content_type == ContentType.TABLE:
            if len(document.content) <= self.chunk_size * 1.5:
                return [document]
        
        # Split the content
        chunks = self.text_splitter.split_text(document.content)
        
        if not chunks:
            return [document]
        
        # Create new documents for each chunk
        chunked_docs = []
        for idx, chunk_content in enumerate(chunks):
            chunk_doc = Document(
                id="",  # Will be auto-generated
                content=chunk_content,
                content_type=document.content_type,
                metadata={
                    **document.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "original_doc_id": document.id
                },
                source_file=document.source_file,
                page_number=document.page_number,
                chunk_index=idx
            )
            chunked_docs.append(chunk_doc)
        
        return chunked_docs
    
    def chunk_text(self, text: str, metadata: dict = None) -> list[Document]:
        """
        Chunk raw text into documents.
        
        Args:
            text: Raw text to chunk
            metadata: Optional metadata to attach to documents
            
        Returns:
            List of Document objects
        """
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for idx, chunk_content in enumerate(chunks):
            doc = Document(
                id="",
                content=chunk_content,
                content_type=ContentType.TEXT,
                metadata={
                    **(metadata or {}),
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                },
                chunk_index=idx
            )
            documents.append(doc)
        
        return documents
    
    def estimate_chunks(self, text: str) -> int:
        """
        Estimate the number of chunks for a given text.
        
        Args:
            text: Text to estimate chunks for
            
        Returns:
            Estimated number of chunks
        """
        if not text:
            return 0
        
        # Simple estimation based on chunk size and overlap
        effective_chunk_size = self.chunk_size - self.chunk_overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = self.chunk_size
        
        return max(1, (len(text) + effective_chunk_size - 1) // effective_chunk_size)

