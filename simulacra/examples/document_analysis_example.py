#!/usr/bin/env python3
"""
Example script for using the DocumentAnalysisAgent to analyze a folder of text documents.

This script demonstrates how to:
1. Initialize the DocumentAnalysisAgent
2. Process a folder of documents
3. Generate and save analysis results
4. Create a human-readable report

Usage:
    python document_analysis_example.py /path/to/document/folder [output_dir]

Dependencies:
    - simulacra package
    - OpenAI API key set as environment variable OPENAI_API_KEY
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Import the DocumentAnalysisAgent directly from the module
from src.document_analysis.document_analyzer import DocumentAnalysisAgent


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze writing style in a folder of documents"
    )
    parser.add_argument(
        "folder_path", 
        help="Path to the folder containing documents to analyze"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./analysis_results",
        help="Directory to save analysis results (default: ./analysis_results)"
    )
    parser.add_argument(
        "--extensions", "-e",
        default=".txt,.md,.rst,.html,.tex,.docx",
        help="Comma-separated list of file extensions to include (default: .txt,.md,.rst,.html,.tex,.docx)"
    )
    parser.add_argument(
        "--max-files", "-m",
        type=int,
        default=50,
        help="Maximum number of files to analyze (default: 50)"
    )
    parser.add_argument(
        "--model", 
        default="gpt-4-turbo",
        help="OpenAI model to use (default: gpt-4-turbo)"
    )
    
    args = parser.parse_args()
    
    # Validate the folder path
    folder_path = Path(args.folder_path)
    if not folder_path.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return 1
    if not folder_path.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse file extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    
    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
    
    print(f"Analyzing documents in: {folder_path}")
    print(f"File extensions: {', '.join(extensions)}")
    print(f"Maximum files: {args.max_files}")
    print(f"OpenAI model: {args.model}")
    print(f"Results will be saved to: {output_dir}")
    print("\n" + "="*50 + "\n")
    
    try:
        # Initialize the DocumentAnalysisAgent
        agent = DocumentAnalysisAgent(
            model=args.model,
            max_files_per_analysis=args.max_files
        )
        
        # Generate a timestamp for unique output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Analyze the documents
        print("Starting document analysis...")
        analysis = agent.analyze_documents(
            folder_path=str(folder_path),
            file_extensions=extensions
        )
        
        # Save the analysis as JSON
        json_output_path = output_dir / f"analysis_{timestamp}.json"
        json_path = agent.save_analysis(analysis, str(json_output_path))
        print(f"Analysis saved to: {json_path}")
        
        # Generate a human-readable report
        report_output_path = output_dir / f"report_{timestamp}.md"
        report_path = agent.generate_analysis_report(analysis, str(report_output_path))
        print(f"Report saved to: {report_path}")
        
        print("\nAnalysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
