#!/usr/bin/env python3
"""
Test script to debug document parsing functionality.
"""

import sys
import os
from pathlib import Path

# Import the DocumentParser
from src.file_operations.document_parser import DocumentParser
from src.file_operations.directory_traversal import DirectoryTraverser

def main():
    # The folder to examine
    folder_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9"
    file_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9/metamorphosis.txt"
    print(f"Testing document parser for: {file_path}")
    
    # Create a DocumentParser and DirectoryTraverser
    parser = DocumentParser()
    traverser = DirectoryTraverser()
    
    # Begin testing
    print("\n1. Testing get_supported_formats:")
    try:
        formats = parser.get_supported_formats()
        print(f"Supported formats: {formats}")
    except Exception as e:
        print(f"Error getting supported formats: {e}")
    
    print("\n2. Testing parse_document:")
    try:
        parsed_doc = parser.parse_document(file_path)
        print(f"Successfully parsed document: {file_path}")
        print(f"Content length: {len(parsed_doc.text_content)} characters")
        print(f"Content preview: {parsed_doc.text_content[:100]}...")
        print(f"Metadata: {parsed_doc.metadata}")
    except Exception as e:
        print(f"Error parsing document: {e}")
    
    print("\n3. Testing full document analysis pipeline:")
    # Find txt files
    try:
        txt_files = traverser.find_files(folder_path, "*.txt", recursive=True)
        print(f"Found {len(txt_files)} txt files")
        
        # Try to parse each one
        for file_meta in txt_files:
            try:
                doc = parser.parse_document(file_meta.path)
                print(f"Successfully parsed: {file_meta.path}")
                print(f"Content length: {len(doc.text_content)} characters")
            except Exception as e:
                print(f"Failed to parse {file_meta.path}: {e}")
    except Exception as e:
        print(f"Error finding txt files: {e}")
            
    print("\n4. Testing direct reading of file:")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Direct read successful. Content length: {len(content)} characters")
            print(f"Content preview: {content[:100]}...")
    except Exception as e:
        print(f"Error reading file directly: {e}")

if __name__ == "__main__":
    main()
