"""
Document parsing tools for Simulacra.

This module provides functionality to parse different document formats (txt, pdf, docx),
with integration with the OpenAI Agents SDK.
"""

import os
import io
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, BinaryIO
from ..openai_agents import AgentTool, NamedAgentTool
from pydantic import BaseModel, Field

class DocumentContent(BaseModel):
    """Content extracted from a document."""
    path: str
    filename: str
    extension: str
    text_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    page_count: Optional[int] = None
    
class DocumentParser:
    """
    A tool for parsing different document formats.
    Integrates with the OpenAI Agents SDK.
    """
    
    def __init__(self, max_file_size: int = 50 * 1024 * 1024):  # 50 MB default
        """
        Initialize the document parser.
        
        Args:
            max_file_size: Maximum file size in bytes to parse
        """
        self.max_file_size = max_file_size
        self._supported_extensions = {'.txt', '.md', '.pdf', '.docx', '.doc', '.rtf'}
        
    def get_agent_tools(self) -> List[NamedAgentTool]:
        """Get the agent tools for document parsing."""
        
        return [
            NamedAgentTool(
                name="parse_document",
                description="Parse a document file (txt, pdf, docx) and extract its content",
                callable=self.parse_document,
            ),
            NamedAgentTool(
                name="get_supported_formats",
                description="Get a list of supported document formats",
                callable=self.get_supported_formats,
            ),
        ]
    
    def parse_document(self, file_path: str) -> DocumentContent:
        """
        Parse a document file and extract its content.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            DocumentContent object with the extracted content
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")
        
        # Check file size
        if file.stat().st_size > self.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed: {file.stat().st_size} > {self.max_file_size}")
        
        # Check if the file extension is supported
        extension = file.suffix.lower()
        if extension not in self._supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")
        
        # Parse based on file extension
        if extension == '.txt' or extension == '.md':
            return self._parse_text_file(file_path)
        elif extension == '.pdf':
            return self._parse_pdf_file(file_path)
        elif extension in ['.docx', '.doc']:
            return self._parse_word_file(file_path)
        elif extension == '.rtf':
            return self._parse_rtf_file(file_path)
        else:
            # This should not happen due to the check above
            raise ValueError(f"Unsupported file extension: {extension}")
    
    def get_supported_formats(self) -> List[str]:
        """
        Get a list of supported document formats.
        
        Returns:
            List of supported document formats (file extensions)
        """
        return list(self._supported_extensions)
    
    def _parse_text_file(self, file_path: str) -> DocumentContent:
        """
        Parse a text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            DocumentContent object with the extracted content
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with Latin-1 encoding as fallback
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception as e:
                raise ValueError(f"Error reading text file {file_path}: {e}")
        
        file = Path(file_path)
        
        # Simple metadata for text files
        line_count = len(content.split('\n'))
        word_count = len(re.findall(r'\w+', content))
        char_count = len(content)
        
        metadata = {
            'line_count': line_count,
            'word_count': word_count,
            'character_count': char_count,
            'content_type': 'text/plain' if file.suffix.lower() == '.txt' else 'text/markdown'
        }
        
        return DocumentContent(
            path=file_path,
            filename=file.name,
            extension=file.suffix.lower(),
            text_content=content,
            metadata=metadata
        )
    
    def _parse_pdf_file(self, file_path: str) -> DocumentContent:
        """
        Parse a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            DocumentContent object with the extracted content
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 is required for PDF parsing. Install it with 'pip install PyPDF2'.")
        
        file = Path(file_path)
        
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                page_count = len(pdf_reader.pages)
                
                # Extract text from all pages
                text_content = ""
                for page_num in range(page_count):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n\n"
                
                # Extract document info as metadata
                metadata = {}
                if pdf_reader.metadata:
                    for key, value in pdf_reader.metadata.items():
                        if key.startswith('/'):
                            key = key[1:]  # Remove leading slash
                        metadata[key] = str(value)
                
                # Add some basic metadata
                metadata['page_count'] = page_count
                word_count = len(re.findall(r'\w+', text_content))
                metadata['word_count'] = word_count
                metadata['content_type'] = 'application/pdf'
                
                return DocumentContent(
                    path=file_path,
                    filename=file.name,
                    extension=file.suffix.lower(),
                    text_content=text_content,
                    metadata=metadata,
                    page_count=page_count
                )
        except Exception as e:
            raise ValueError(f"Error parsing PDF file {file_path}: {e}")
    
    def _parse_word_file(self, file_path: str) -> DocumentContent:
        """
        Parse a Word document.
        
        Args:
            file_path: Path to the Word document
            
        Returns:
            DocumentContent object with the extracted content
        """
        try:
            import docx
        except ImportError:
            raise ImportError("python-docx is required for Word document parsing. Install it with 'pip install python-docx'.")
        
        file = Path(file_path)
        
        try:
            # For .doc files, we'll need to convert or use a different approach
            if file.suffix.lower() == '.doc':
                return self._parse_doc_file(file_path)
            
            # Parse .docx file
            doc = docx.Document(file_path)
            
            # Extract text
            paragraphs = [para.text for para in doc.paragraphs]
            text_content = '\n'.join(paragraphs)
            
            # Extract metadata
            metadata = {
                'paragraph_count': len(paragraphs),
                'word_count': len(re.findall(r'\w+', text_content)),
                'content_type': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            }
            
            # Count pages (approximate)
            page_count = len(doc.sections)
            
            return DocumentContent(
                path=file_path,
                filename=file.name,
                extension=file.suffix.lower(),
                text_content=text_content,
                metadata=metadata,
                page_count=page_count
            )
        except Exception as e:
            raise ValueError(f"Error parsing Word document {file_path}: {e}")
    
    def _parse_doc_file(self, file_path: str) -> DocumentContent:
        """
        Parse a legacy .doc Word document.
        
        Args:
            file_path: Path to the .doc file
            
        Returns:
            DocumentContent object with the extracted content
        """
        try:
            # First try with antiword if available
            content = self._try_antiword(file_path)
            
            # Fallback to textract if antiword fails
            if content is None:
                content = self._try_textract(file_path)
            
            # Last resort: try python-docx2txt
            if content is None:
                content = self._try_docx2txt(file_path)
            
            # If all fail, raise an error
            if content is None:
                raise ImportError("Could not parse .doc file. Install 'antiword', 'textract', or 'docx2txt'.")
            
            file = Path(file_path)
            
            # Basic metadata
            metadata = {
                'word_count': len(re.findall(r'\w+', content)),
                'content_type': 'application/msword'
            }
            
            return DocumentContent(
                path=file_path,
                filename=file.name,
                extension=file.suffix.lower(),
                text_content=content,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Error parsing .doc file {file_path}: {e}")
    
    def _try_antiword(self, file_path: str) -> Optional[str]:
        """Try to extract text using antiword."""
        try:
            import subprocess
            result = subprocess.run(['antiword', file_path], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout
        except Exception:
            pass
        return None
    
    def _try_textract(self, file_path: str) -> Optional[str]:
        """Try to extract text using textract."""
        try:
            import textract
            return textract.process(file_path).decode('utf-8')
        except Exception:
            pass
        return None
    
    def _try_docx2txt(self, file_path: str) -> Optional[str]:
        """Try to extract text using docx2txt."""
        try:
            import docx2txt
            return docx2txt.process(file_path)
        except Exception:
            pass
        return None
    
    def _parse_rtf_file(self, file_path: str) -> DocumentContent:
        """
        Parse a RTF file.
        
        Args:
            file_path: Path to the RTF file
            
        Returns:
            DocumentContent object with the extracted content
        """
        try:
            import striprtf
        except ImportError:
            raise ImportError("striprtf is required for RTF parsing. Install it with 'pip install striprtf'.")
        
        file = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                rtf_text = f.read()
            
            # Strip RTF formatting
            text_content = striprtf.rtf_to_text(rtf_text)
            
            # Simple metadata for RTF files
            word_count = len(re.findall(r'\w+', text_content))
            
            metadata = {
                'word_count': word_count,
                'content_type': 'application/rtf'
            }
            
            return DocumentContent(
                path=file_path,
                filename=file.name,
                extension=file.suffix.lower(),
                text_content=text_content,
                metadata=metadata
            )
        except Exception as e:
            raise ValueError(f"Error parsing RTF file {file_path}: {e}")
