# services/document_processor_factory.py
"""Factory for creating document processor instances"""
from typing import Dict, Type
from core.interfaces import IDocumentProcessor
from infrastructure.document_processors import PDFProcessor
# To add DOCX support in the future, you would:
# 1. Create a DOCXProcessor in infrastructure/document_processors.py
# 2. Import it here: from infrastructure.document_processors import DOCXProcessor

class DocumentProcessorFactory:
    """Factory to get the correct document processor based on file type"""
    
    def __init__(self):
        self._processors: Dict[str, Type[IDocumentProcessor]] = {
            "pdf": PDFProcessor,
            # "docx": DOCXProcessor, # Add this line to support DOCX files
        }

    def get_processor(self, file_type: str) -> IDocumentProcessor:
        """
        Returns an instance of the appropriate document processor.
        
        Args:
            file_type (str): The file extension (e.g., "pdf", "docx").
            
        Returns:
            IDocumentProcessor: An instance of a class that implements the interface.
            
        Raises:
            ValueError: If the file type is not supported.
        """
        processor_class = self._processors.get(file_type.lower())
        if not processor_class:
            raise ValueError(f"Unsupported file type: '{file_type}'")
        
        # We return a new instance each time
        return processor_class()