"""
Directory traversal tools for Simulacra.

This module provides functionality to traverse directories and read document files,
with integration with the OpenAI Agents SDK.
"""

import os
import fnmatch
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Set, Tuple
from ..openai_agents import AgentTool, NamedAgentTool
from pydantic import BaseModel, Field

class FileMetadata(BaseModel):
    """Metadata for a file in the directory."""
    path: str
    name: str
    extension: str
    size: int
    modified: float
    created: float
    is_binary: bool = False
    
class DirectoryContents(BaseModel):
    """Contents of a directory."""
    directory_path: str
    files: List[FileMetadata]
    subdirectories: List[str]
    
class DirectoryTraverser:
    """
    A tool for traversing directories and reading document files.
    Integrates with the OpenAI Agents SDK.
    """
    
    def __init__(self, 
                 max_file_size: int = 10 * 1024 * 1024,  # 10 MB default max file size
                 excluded_dirs: Set[str] = None):
        """
        Initialize the directory traverser.
        
        Args:
            max_file_size: Maximum file size in bytes to read
            excluded_dirs: Directories to exclude from traversal (e.g., .git, node_modules)
        """
        self.max_file_size = max_file_size
        self.excluded_dirs = excluded_dirs or {'.git', 'node_modules', '__pycache__', 'venv', '.env'}
        self._visited_dirs: Set[str] = set()
        
    def get_agent_tools(self) -> List[NamedAgentTool]:
        """Get the agent tools for directory traversal."""
        
        return [
            NamedAgentTool(
                name="list_directory",
                description="List contents of a directory",
                callable=self.list_directory,
            ),
            NamedAgentTool(
                name="find_files",
                description="Find files matching a pattern",
                callable=self.find_files,
            ),
            NamedAgentTool(
                name="read_file",
                description="Read the contents of a file",
                callable=self.read_file,
            ),
        ]
    
    def list_directory(self, directory_path: str) -> DirectoryContents:
        """
        List contents of a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            DirectoryContents object with files and subdirectories
        """
        # Ensure the path exists and is a directory
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not directory.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {directory_path}")
        
        # Track visited directories to avoid cycles
        abs_path = directory.resolve()
        if str(abs_path) in self._visited_dirs:
            return DirectoryContents(
                directory_path=directory_path,
                files=[],
                subdirectories=[]
            )
        self._visited_dirs.add(str(abs_path))
        
        files = []
        subdirectories = []
        
        try:
            for item in directory.iterdir():
                if item.is_dir():
                    # Skip excluded directories
                    if item.name in self.excluded_dirs:
                        continue
                    subdirectories.append(str(item))
                elif item.is_file():
                    # Skip files larger than max_file_size
                    if item.stat().st_size > self.max_file_size:
                        continue
                    
                    # Add file metadata
                    file_stat = item.stat()
                    is_binary = self._is_binary_file(item)
                    
                    files.append(FileMetadata(
                        path=str(item),
                        name=item.name,
                        extension=item.suffix.lower(),
                        size=file_stat.st_size,
                        modified=file_stat.st_mtime,
                        created=file_stat.st_ctime,
                        is_binary=is_binary
                    ))
        except PermissionError as e:
            print(f"Permission error accessing {directory_path}: {e}")
        
        return DirectoryContents(
            directory_path=directory_path,
            files=files,
            subdirectories=subdirectories
        )
    
    def find_files(self, 
                 directory_path: str, 
                 pattern: str = "*", 
                 recursive: bool = True,
                 include_binary: bool = False) -> List[FileMetadata]:
        """
        Find files matching a pattern in a directory.
        
        Args:
            directory_path: Path to the directory
            pattern: File pattern to match (supports wildcards)
            recursive: Whether to search recursively
            include_binary: Whether to include binary files
            
        Returns:
            List of FileMetadata objects for matching files
        """
        # Clear visited directories
        self._visited_dirs = set()
        
        matching_files = []
        directories_to_process = [directory_path]
        
        while directories_to_process:
            current_dir = directories_to_process.pop(0)
            
            try:
                dir_contents = self.list_directory(current_dir)
                
                # Add matching files
                for file in dir_contents.files:
                    if not include_binary and file.is_binary:
                        continue
                        
                    if fnmatch.fnmatch(file.name, pattern):
                        matching_files.append(file)
                
                # Add subdirectories if recursive
                if recursive:
                    directories_to_process.extend(dir_contents.subdirectories)
            except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
                print(f"Error processing directory {current_dir}: {e}")
        
        return matching_files
    
    def read_file(self, file_path: str) -> str:
        """
        Read the contents of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File contents as string
        """
        file = Path(file_path)
        if not file.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not file.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")
        
        # Check file size
        if file.stat().st_size > self.max_file_size:
            raise ValueError(f"File size exceeds maximum allowed: {file.stat().st_size} > {self.max_file_size}")
        
        # Check if file is binary
        if self._is_binary_file(file):
            raise ValueError(f"File appears to be binary: {file_path}. Use a document parser for binary files.")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try reading with Latin-1 encoding as fallback
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"Error reading file {file_path}: {e}")
    
    def walk_directory(self, directory_path: str) -> Iterator[Tuple[str, List[str], List[FileMetadata]]]:
        """
        Walk a directory recursively, similar to os.walk but with file metadata.
        
        Args:
            directory_path: Path to the directory
            
        Yields:
            Tuples of (current_dir, subdirectories, files)
        """
        # Clear visited directories
        self._visited_dirs = set()
        
        directories_to_process = [directory_path]
        
        while directories_to_process:
            current_dir = directories_to_process.pop(0)
            
            try:
                dir_contents = self.list_directory(current_dir)
                
                # Add subdirectories to process queue
                directories_to_process.extend(dir_contents.subdirectories)
                
                # Extract just the directory names for the subdirectory list
                subdir_names = [Path(path).name for path in dir_contents.subdirectories]
                
                yield current_dir, subdir_names, dir_contents.files
                
            except (FileNotFoundError, NotADirectoryError, PermissionError) as e:
                print(f"Error processing directory {current_dir}: {e}")
    
    def _is_binary_file(self, file_path: Path) -> bool:
        """
        Check if a file is binary.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the file is binary, False otherwise
        """
        # Quick check based on file extension for common binary formats
        binary_extensions = {
            '.pdf', '.docx', '.pptx', '.xlsx', '.zip', '.tar', '.gz', '.jpg', '.jpeg', 
            '.png', '.gif', '.bmp', '.mp3', '.mp4', '.avi', '.mkv', '.exe', '.dll'
        }
        
        if file_path.suffix.lower() in binary_extensions:
            return True
        
        # Check file content (read a sample)
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                if b'\x00' in sample:
                    return True
                
                # Count text vs binary characters
                text_chars = set(bytes(range(32, 127)) + b'\r\n\t\b')
                return len(sample) > 0 and sum(c in text_chars for c in sample) / len(sample) < 0.7
        except Exception:
            # If we can't read the file, assume it's binary to be safe
            return True
