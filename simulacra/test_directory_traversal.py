#!/usr/bin/env python3
"""
Test script to debug directory traversal and file finding.
"""

import sys
import os
from pathlib import Path

# Import the DirectoryTraverser
from src.file_operations.directory_traversal import DirectoryTraverser

def main():
    # The folder to examine
    folder_path = "/Users/danielkliewer/simulacra03/simulacra-web/uploads/3b7bca95-c2ec-43a8-bfac-534995f7b1d9"
    print(f"Testing directory traversal for: {folder_path}")
    
    # Create a DirectoryTraverser
    traverser = DirectoryTraverser()
    
    # List directory contents
    try:
        print("\n1. Testing list_directory:")
        dir_contents = traverser.list_directory(folder_path)
        print(f"Found {len(dir_contents.files)} files and {len(dir_contents.subdirectories)} subdirectories")
        for file in dir_contents.files:
            print(f"  - File: {file.name} ({file.extension})")
        for subdir in dir_contents.subdirectories:
            print(f"  - Subdir: {os.path.basename(subdir)}")
    except Exception as e:
        print(f"Error listing directory: {e}")
    
    # Find files with various patterns
    extensions = ['.txt', '.md', '.rst', '.html', '.tex', '.docx']
    for ext in extensions:
        try:
            pattern = f"*{ext}"
            print(f"\n2. Testing find_files with pattern: {pattern}")
            files = traverser.find_files(folder_path, pattern, recursive=True)
            print(f"Found {len(files)} files matching {pattern}")
            for file in files:
                print(f"  - {file.path}")
        except Exception as e:
            print(f"Error finding files with {pattern}: {e}")
    
    # Direct file check using Path
    print("\n3. Direct file check using Path:")
    folder = Path(folder_path)
    for ext in extensions:
        files = list(folder.glob(f"**/*{ext}"))
        print(f"Path.glob found {len(files)} files with extension {ext}")
        for file in files:
            print(f"  - {file}")
    
    # OS walk check
    print("\n4. OS walk check:")
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            print(f"  - {os.path.join(root, file)}")

if __name__ == "__main__":
    main()
