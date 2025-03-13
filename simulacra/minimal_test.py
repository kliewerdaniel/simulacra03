#!/usr/bin/env python3
"""
Minimal test script for DocumentAnalysisAgent with additional standard library checks
"""
import os
import sys
from pathlib import Path

from src.document_analysis.document_analyzer import DocumentAnalysisAgent
from src.file_operations.directory_traversal import DirectoryTraverser, FileMetadata
from src.file_operations.document_parser import DocumentParser

def main():
    # Set the folder path to analyze
    folder_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9"
    file_path = os.path.join(folder_path, "metamorphosis.txt")
    
    # Check using standard Python file operations
    print(f"Standard Python checks:")
    print(f"os.path.exists('{folder_path}'): {os.path.exists(folder_path)}")
    print(f"os.path.isdir('{folder_path}'): {os.path.isdir(folder_path)}")
    print(f"os.path.exists('{file_path}'): {os.path.exists(file_path)}")
    print(f"os.path.isfile('{file_path}'): {os.path.isfile(file_path)}")
    
    # List files using standard Python
    print("\nFiles in directory using os.listdir:")
    try:
        files = os.listdir(folder_path)
        print(f"Found {len(files)} files/directories:")
        for file in files:
            full_path = os.path.join(folder_path, file)
            print(f"  - {file} ({'directory' if os.path.isdir(full_path) else 'file'})")
    except Exception as e:
        print(f"Error using os.listdir: {e}")
    
    # Now create a DirectoryTraverser directly and test
    print("\nDirect DirectoryTraverser test:")
    traverser = DirectoryTraverser()
    
    # Test if any exclusion rules might be preventing the file from being found
    print(f"DirectoryTraverser excluded_dirs: {traverser.excluded_dirs}")
    
    # Try listing directory
    try:
        print(f"\nListing files with DirectoryTraverser.list_directory:")
        dir_contents = traverser.list_directory(folder_path)
        print(f"Files in directory: {len(dir_contents.files)}")
        for file in dir_contents.files:
            print(f"  - {file.name} ({file.extension})")
    except Exception as e:
        print(f"Error using DirectoryTraverser.list_directory: {e}")
    
    # Try finding files with pattern
    try:
        print(f"\nFinding files with DirectoryTraverser.find_files:")
        matching_files = traverser.find_files(folder_path, "*.txt", recursive=True, include_binary=True)
        print(f"Found {len(matching_files)} files matching '*.txt':")
        for file in matching_files:
            print(f"  - {file.path}")
    except Exception as e:
        print(f"Error using DirectoryTraverser.find_files: {e}")
    
    # Try creating and using DocumentAnalysisAgent
    print("\nTesting DocumentAnalysisAgent:")
    agent = DocumentAnalysisAgent()
    
    # Try reading documents with the read_documents method but adding more debug
    print("\nDebugging agent.read_documents():")
    try:
        # Examine the implementation
        print(f"Step 1: DirectoryTraverser finds files correctly")
        matching_files = traverser.find_files(folder_path, "*.txt", recursive=True)
        print(f"DirectoryTraverser found {len(matching_files)} files matching '*.txt'")
        
        print(f"\nStep 2: Try to parse one file directly with DocumentParser")
        parser = DocumentParser()
        file_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9/metamorphosis.txt"
        try:
            doc_content = parser.parse_document(file_path)
            print(f"DocumentParser successfully parsed file: {len(doc_content.text_content)} characters")
        except Exception as e:
            print(f"DocumentParser error: {e}")
        
        print(f"\nStep 3: Try read_documents with direct debugging inside")
        documents = []
        file_extensions = ['.txt']
        file_patterns = [f"*{ext}" if ext.startswith('.') else f"*.{ext}" for ext in file_extensions]
        print(f"File patterns: {file_patterns}")
        
        for pattern in file_patterns:
            print(f"Processing pattern: {pattern}")
            matching_files = agent.directory_traverser.find_files(
                directory_path=folder_path,
                pattern=pattern,
                recursive=True,
                include_binary=True
            )
            print(f"Found {len(matching_files)} files matching {pattern}")
            
            for file_metadata in matching_files:
                print(f"Processing file: {file_metadata.path}")
                try:
                    # Use DocumentParser to parse the file
                    doc_content = agent.document_parser.parse_document(file_metadata.path)
                    print(f"Successfully parsed: {len(doc_content.text_content)} characters")
                    
                    # Get relative path for consistent references
                    try:
                        relative_path = os.path.relpath(file_metadata.path, folder_path)
                    except ValueError:
                        # Fallback if paths are on different drives
                        relative_path = file_metadata.path
                    
                    documents.append({
                        'path': relative_path,
                        'name': file_metadata.name,
                        'content': doc_content.text_content,
                        'size': file_metadata.size,
                        'modified': file_metadata.modified,
                        'created': file_metadata.created,
                        'metadata': doc_content.metadata
                    })
                    print(f"Added document to results")
                except Exception as e:
                    print(f"Error parsing file {file_metadata.path}: {e}")
        
        print(f"\nManual implementation read {len(documents)} documents")
        for doc in documents:
            print(f"  - {doc['name']} ({len(doc['content'])} characters)")
        
        print(f"\nNow compare with agent.read_documents():")
        agent_documents = agent.read_documents(folder_path, ['.txt'])
        print(f"Agent read_documents found: {len(agent_documents)} documents")
        
    except Exception as e:
        print(f"Error in debugging: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
