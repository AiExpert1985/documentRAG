# utils/helpers.py
"""Utility helper functions"""
import hashlib
import re

def get_file_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()


# Utility functions
def validate_document_id(doc_id: str) -> bool:
    uuid_pattern = r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$'
    return bool(re.match(uuid_pattern, doc_id, re.IGNORECASE))