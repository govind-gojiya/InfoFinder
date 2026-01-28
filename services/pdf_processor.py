"""PDF processing service for extracting text, images, and tables."""

import io
import base64
from pathlib import Path
from typing import Generator
import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from models.schemas import Document, ContentType


class PDFProcessor:
    """Process PDF files to extract text, images, and tables."""
    
    def __init__(self):
        self.supported_extensions = {".pdf"}
    
    def process_file(self, file_path: str | Path, source_name: str = None) -> list[Document]:
        """
        Process a PDF file and extract all content.
        
        Args:
            file_path: Path to the PDF file
            source_name: Optional name for the source file
            
        Returns:
            List of Document objects containing extracted content
        """
        file_path = Path(file_path)
        source_name = source_name or file_path.name
        
        if file_path.suffix.lower() not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
        
        documents = []
        
        # Extract text and images using PyMuPDF
        documents.extend(self._extract_text_and_images(file_path, source_name))
        
        # Extract tables using pdfplumber
        documents.extend(self._extract_tables(file_path, source_name))
        
        return documents
    
    def process_uploaded_file(self, uploaded_file, source_name: str = None) -> list[Document]:
        """
        Process an uploaded file (Streamlit UploadedFile).
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            source_name: Optional name for the source file
            
        Returns:
            List of Document objects containing extracted content
        """
        source_name = source_name or uploaded_file.name
        file_bytes = uploaded_file.read()
        
        documents = []
        
        # Extract text and images
        documents.extend(self._extract_text_and_images_from_bytes(file_bytes, source_name))
        
        # Extract tables
        documents.extend(self._extract_tables_from_bytes(file_bytes, source_name))
        
        return documents
    
    def _extract_text_and_images(self, file_path: Path, source_name: str) -> list[Document]:
        """Extract text and images from PDF using PyMuPDF."""
        documents = []
        
        with fitz.open(file_path) as pdf:
            for page_num, page in enumerate(pdf, start=1):
                # Extract text
                text = page.get_text("text").strip()
                if text:
                    doc = Document(
                        id="",
                        content=text,
                        content_type=ContentType.TEXT,
                        metadata={"extraction_method": "pymupdf"},
                        source_file=source_name,
                        page_number=page_num,
                        chunk_index=0
                    )
                    documents.append(doc)
                
                # Extract images
                images = self._extract_page_images(page, page_num, source_name)
                documents.extend(images)
        
        return documents
    
    def _extract_text_and_images_from_bytes(self, file_bytes: bytes, source_name: str) -> list[Document]:
        """Extract text and images from PDF bytes."""
        documents = []
        
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for page_num, page in enumerate(pdf, start=1):
                # Extract text
                text = page.get_text("text").strip()
                if text:
                    doc = Document(
                        id="",
                        content=text,
                        content_type=ContentType.TEXT,
                        metadata={"extraction_method": "pymupdf"},
                        source_file=source_name,
                        page_number=page_num,
                        chunk_index=0
                    )
                    documents.append(doc)
                
                # Extract images
                images = self._extract_page_images(page, page_num, source_name)
                documents.extend(images)
        
        return documents
    
    def _extract_page_images(self, page, page_num: int, source_name: str) -> list[Document]:
        """Extract images from a PDF page."""
        documents = []
        image_list = page.get_images()
        
        for img_index, img_info in enumerate(image_list):
            try:
                xref = img_info[0]
                base_image = page.parent.extract_image(xref)
                
                if base_image:
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Convert to base64 for storage
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    
                    # Create description for the image
                    image_description = f"[Image from {source_name}, page {page_num}, image {img_index + 1}]"
                    
                    doc = Document(
                        id="",
                        content=image_description,
                        content_type=ContentType.IMAGE,
                        metadata={
                            "image_data": image_b64,
                            "image_format": image_ext,
                            "extraction_method": "pymupdf"
                        },
                        source_file=source_name,
                        page_number=page_num,
                        chunk_index=img_index
                    )
                    documents.append(doc)
            except Exception as e:
                # Skip problematic images
                continue
        
        return documents
    
    def _extract_tables(self, file_path: Path, source_name: str) -> list[Document]:
        """Extract tables from PDF using pdfplumber."""
        documents = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    
                    for table_index, table in enumerate(tables):
                        if table:
                            table_text = self._table_to_markdown(table)
                            if table_text.strip():
                                doc = Document(
                                    id="",
                                    content=table_text,
                                    content_type=ContentType.TABLE,
                                    metadata={
                                        "extraction_method": "pdfplumber",
                                        "table_index": table_index
                                    },
                                    source_file=source_name,
                                    page_number=page_num,
                                    chunk_index=table_index
                                )
                                documents.append(doc)
        except Exception as e:
            # If table extraction fails, continue without tables
            pass
        
        return documents
    
    def _extract_tables_from_bytes(self, file_bytes: bytes, source_name: str) -> list[Document]:
        """Extract tables from PDF bytes using pdfplumber."""
        documents = []
        
        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    tables = page.extract_tables()
                    
                    for table_index, table in enumerate(tables):
                        if table:
                            table_text = self._table_to_markdown(table)
                            if table_text.strip():
                                doc = Document(
                                    id="",
                                    content=table_text,
                                    content_type=ContentType.TABLE,
                                    metadata={
                                        "extraction_method": "pdfplumber",
                                        "table_index": table_index
                                    },
                                    source_file=source_name,
                                    page_number=page_num,
                                    chunk_index=table_index
                                )
                                documents.append(doc)
        except Exception as e:
            # If table extraction fails, continue without tables
            pass
        
        return documents
    
    def _table_to_markdown(self, table: list[list]) -> str:
        """Convert a table to markdown format."""
        if not table or not table[0]:
            return ""
        
        lines = []
        
        # Header row
        header = table[0]
        header_str = "| " + " | ".join(str(cell) if cell else "" for cell in header) + " |"
        lines.append(header_str)
        
        # Separator
        separator = "| " + " | ".join("---" for _ in header) + " |"
        lines.append(separator)
        
        # Data rows
        for row in table[1:]:
            if row:
                row_str = "| " + " | ".join(str(cell) if cell else "" for cell in row) + " |"
                lines.append(row_str)
        
        return "\n".join(lines)
    
    def get_page_count(self, file_path: str | Path) -> int:
        """Get the number of pages in a PDF."""
        with fitz.open(file_path) as pdf:
            return len(pdf)
    
    def get_page_count_from_bytes(self, file_bytes: bytes) -> int:
        """Get the number of pages from PDF bytes."""
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            return len(pdf)

