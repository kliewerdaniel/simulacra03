"""
Example demonstrating the use of file operations tools.

This example shows how to:
1. Traverse directories and read document files
2. Parse different document formats (txt, pdf, docx)
3. Save a generated persona to disk in different formats
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path to ensure imports work correctly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.file_operations.directory_traversal import DirectoryTraverser
from src.file_operations.document_parser import DocumentParser
from src.file_operations.persona_serializer import PersonaSerializer
from src.persona_generator.persona import Persona

def demonstrate_directory_traversal(start_dir: str = "./"):
    """Demonstrate the use of the DirectoryTraverser."""
    print("\n=== Directory Traversal Tool Demo ===\n")
    
    traverser = DirectoryTraverser()
    
    # List directory contents
    print(f"Listing contents of {start_dir}:")
    contents = traverser.list_directory(start_dir)
    print(f"Found {len(contents.files)} files and {len(contents.subdirectories)} subdirectories")
    
    # Print first 5 files (if available)
    print("\nSample files:")
    for i, file in enumerate(contents.files[:5]):
        print(f"  {i+1}. {file.name} ({file.extension}, {file.size} bytes)")
    
    # Find text files
    print("\nFinding all text files:")
    text_files = traverser.find_files(start_dir, "*.txt", recursive=True)
    print(f"Found {len(text_files)} text files")
    
    # Read the first text file if available
    if text_files:
        print("\nReading first text file:")
        try:
            file_content = traverser.read_file(text_files[0].path)
            print(f"Read {len(file_content)} characters from {text_files[0].name}")
            print(f"First 100 characters: {file_content[:100]}...")
        except Exception as e:
            print(f"Error reading file: {e}")
    
    return traverser

def demonstrate_document_parser(document_path: str = None):
    """Demonstrate the use of the DocumentParser."""
    print("\n=== Document Parser Tool Demo ===\n")
    
    parser = DocumentParser()
    
    # Show supported formats
    supported_formats = parser.get_supported_formats()
    print(f"Supported formats: {', '.join(supported_formats)}")
    
    # Try to parse a document if a path is provided
    if document_path and os.path.exists(document_path):
        try:
            print(f"\nParsing document: {document_path}")
            doc_content = parser.parse_document(document_path)
            
            print(f"Document: {doc_content.filename}")
            print(f"Extension: {doc_content.extension}")
            print(f"Size: {len(doc_content.text_content)} characters")
            print(f"Metadata: {doc_content.metadata}")
            
            # Show a sample of the content
            preview_length = min(200, len(doc_content.text_content))
            print(f"\nContent preview:\n{doc_content.text_content[:preview_length]}...")
            
        except Exception as e:
            print(f"Error parsing document: {e}")
    else:
        print("\nNo document path provided or file not found. Skipping document parsing demonstration.")
    
    return parser

def demonstrate_persona_serialization():
    """Demonstrate the use of the PersonaSerializer."""
    print("\n=== Persona Serializer Tool Demo ===\n")
    
    # Create a test persona
    test_persona = Persona(
        name="Example Writer",
        traits=["analytical", "precise", "verbose", "technical"],
        background="Computer science professor with 15 years of experience in AI research.",
        communication_style="Direct and technical, with a focus on precision and clarity.",
        knowledge_areas=["Artificial Intelligence", "Machine Learning", "Computer Science", "Research Methods"],
        additional_details={
            "writing_voice_summary": "Formal and structured, with a clear logical progression and technical vocabulary.",
            "vocabulary_profile": {
                "complexity": "high",
                "technicality": "high",
                "variety": "above average"
            },
            "rhetorical_devices": ["analogies", "logical frameworks", "empirical evidence"]
        }
    )
    
    serializer = PersonaSerializer(default_output_dir="./output")
    
    # Get supported formats
    supported_formats = serializer.get_supported_formats()
    print(f"Supported formats: {', '.join(supported_formats)}")
    
    # Create output directory if it doesn't exist
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Demonstrate saving in different formats
    results = {}
    
    print("\nSaving persona in different formats:")
    for format in ["json", "yaml", "markdown"]:
        try:
            output_path = os.path.join(output_dir, f"example_persona.{format}")
            result = serializer.save_persona(
                persona=test_persona,
                output_path=output_path,
                format=format,
                overwrite=True
            )
            
            results[format] = result
            print(f"  - Saved as {format}: {result.path} ({result.size} bytes)")
        except Exception as e:
            print(f"  - Error saving as {format}: {e}")
    
    # Try to load a persona from the JSON file
    try:
        if "json" in results and results["json"].success:
            print("\nLoading persona from JSON file:")
            loaded_persona = serializer.load_persona(results["json"].path)
            print(f"Successfully loaded persona: {loaded_persona.name}")
            print(f"Traits: {', '.join(loaded_persona.traits)}")
    except Exception as e:
        print(f"Error loading persona: {e}")
    
    return serializer, results

def main():
    print("File Operations Tools Demo")
    print("=========================")
    
    # Get the examples directory as a starting point
    examples_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Demonstrate directory traversal
    traverser = demonstrate_directory_traversal(examples_dir)
    
    # Look for a sample document to parse
    sample_docs = []
    # Try to find some sample documents to parse
    for ext in [".txt", ".md", ".pdf", ".docx"]:
        files = traverser.find_files(examples_dir, f"*{ext}", recursive=False)
        if files:
            sample_docs.extend(files)
    
    # Demonstrate document parser with a sample document if found
    sample_doc_path = sample_docs[0].path if sample_docs else None
    parser = demonstrate_document_parser(sample_doc_path)
    
    # Demonstrate persona serialization
    serializer, results = demonstrate_persona_serialization()
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()
