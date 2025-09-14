# utils/helpers.py
"""Utility helper functions"""
import hashlib

def get_file_hash(content: bytes) -> str:
    """Calculates the SHA256 hash of file content."""
    return hashlib.sha256(content).hexdigest()