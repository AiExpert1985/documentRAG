# infrastructure/document_processors.py
"""Document processing implementations"""
import asyncio
import uuid
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredPDFLoader

from core.interfaces import IDocumentProcessor, Chunk

class PDFProcessor(IDocumentProcessor):
    """PDF document processor using Unstructured"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    async def process(self, file_path: str, file_type: str) -> List[Chunk]:
        """Process PDF and return chunks"""
        if file_type.lower() != 'pdf':
            raise ValueError(f"PDFProcessor cannot handle {file_type} files")
        
        loader = UnstructuredPDFLoader(
            file_path,
            mode="elements",
            strategy="hi_res"
        )
        
        # Load document
        documents = await asyncio.to_thread(loader.load)
        if not documents:
            raise ValueError("No content extracted from PDF")
        
        # Split into chunks
        split_docs = await asyncio.to_thread(
            self.text_splitter.split_documents,
            documents
        )
        
        # Convert to domain model
        chunks = []
        for i, doc in enumerate(split_docs):
            chunk_id = f"{uuid.uuid4()}_{i}"
            chunks.append(Chunk(
                id=chunk_id,
                content=doc.page_content,
                document_id="",  # Will be set by service
                metadata={
                    "page": doc.metadata.get("page", -1) + 1,
                    "source": doc.metadata.get("source", "")
                }
            ))
        
        return chunks
    
    async def validate(self, file_path: str, file_type: str) -> bool:
        """Validate PDF file"""
        if file_type.lower() != 'pdf':
            return False
        
        try:
            # Basic validation - try to open the file
            with open(file_path, 'rb') as f:
                header = f.read(4)
                return header == b'%PDF'
        except Exception:
            return False