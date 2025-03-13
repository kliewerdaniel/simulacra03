#!/usr/bin/env python3
"""
Simplified example to directly analyze documents without using DocumentAnalysisAgent.
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Import the directly needed classes
from src.file_operations.directory_traversal import DirectoryTraverser
from src.file_operations.document_parser import DocumentParser

def main():
    # The folder to examine
    folder_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9"
    print(f"Analyzing documents in: {folder_path}")
    
    # Create DirectoryTraverser and DocumentParser
    traverser = DirectoryTraverser()
    parser = DocumentParser()
    
    # List all files in the directory
    dir_contents = traverser.list_directory(folder_path)
    print(f"Directory contents: {len(dir_contents.files)} files")
    for file in dir_contents.files:
        print(f"  - {file.name} ({file.extension})")
    
    # Convert extensions format as in DocumentAnalysisAgent.read_documents
    extensions = ['.txt', '.md']
    file_patterns = [f"*{ext}" if ext.startswith('.') else f"*.{ext}" for ext in extensions]
    
    # Find files matching extensions
    documents = []
    for pattern in file_patterns:
        print(f"\nSearching for files matching: {pattern}")
        matching_files = traverser.find_files(
            directory_path=folder_path,
            pattern=pattern,
            recursive=True,
            include_binary=True
        )
        
        print(f"Found {len(matching_files)} files matching {pattern}")
        for file_metadata in matching_files:
            try:
                # Parse each file
                doc_content = parser.parse_document(file_metadata.path)
                print(f"Successfully parsed: {file_metadata.path}")
                print(f"  Content length: {len(doc_content.text_content)} characters")
                
                # Add to documents list as DocumentAnalysisAgent would
                documents.append({
                    'path': file_metadata.path,
                    'name': file_metadata.name,
                    'content': doc_content.text_content,
                    'size': file_metadata.size,
                    'modified': file_metadata.modified,
                    'created': file_metadata.created,
                    'metadata': doc_content.metadata
                })
            except Exception as e:
                print(f"Error parsing file {file_metadata.path}: {e}")
    
    # Print summary
    print(f"\nTotal documents found and parsed: {len(documents)}")
    for doc in documents:
        print(f"  - {doc['name']} ({len(doc['content'])} chars)")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
