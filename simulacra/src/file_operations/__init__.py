"""
File operations module for Simulacra.

This module provides tools for:
1. Directory traversal and document reading
2. Document format parsing (txt, pdf, docx)
3. Persona serialization and storage
"""

from .directory_traversal import DirectoryTraverser
from .document_parser import DocumentParser 
from .persona_serializer import PersonaSerializer

__all__ = [
    'DirectoryTraverser',
    'DocumentParser',
    'PersonaSerializer',
]
