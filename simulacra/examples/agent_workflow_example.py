#!/usr/bin/env python3
"""
Example script demonstrating the Agent Workflow with handoffs between agents.

This script shows how to:
1. Set up the agent workflow
2. Process a folder of documents using handoffs between agents
3. Handle the final results returned to the user

Usage:
    python agent_workflow_example.py /path/to/document/folder [--output-dir OUTPUT_DIR] [--trace-dir TRACE_DIR]

Dependencies:
    - simulacra package
    - OpenAI API key set as environment variable OPENAI_API_KEY
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add the parent directory to the path so we can import the simulacra package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the AgentWorkflow
from src.agent_workflow import AgentWorkflow


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Analyze documents and generate author personas using agent workflow with handoffs"
    )
    parser.add_argument(
        "folder_path",
        help="Path to the folder containing documents to analyze"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./output",
        help="Directory to save output files (default: ./output)"
    )
    parser.add_argument(
        "--trace-dir", "-t",
        default=None,
        help="Directory to save trace logs (default: None, tracing disabled)"
    )
    parser.add_argument(
        "--extensions", "-e",
        default=".txt,.md,.rst,.html,.tex,.docx",
        help="Comma-separated list of file extensions to include"
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

    # Parse file extensions
    extensions = [ext.strip() for ext in args.extensions.split(",")]
    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trace directory if requested
    trace_dir = None
    if args.trace_dir:
        trace_dir = Path(args.trace_dir) / f"trace_{timestamp}"
        trace_dir.mkdir(parents=True, exist_ok=True)

    # Set up the workflow
    try:
        print(f"Initializing agent workflow...")
        print(f"Using model: {args.model}")
        print(f"Output directory: {output_dir}")
        if trace_dir:
            print(f"Trace directory: {trace_dir}")
        print("\n" + "="*50 + "\n")

        # Create the agent workflow
        workflow = AgentWorkflow(
            model=args.model,
            output_dir=str(output_dir),
            trace_dir=str(trace_dir) if trace_dir else None
        )

        # Display information about the process
        print(f"Analyzing documents in: {folder_path}")
        print(f"File extensions: {', '.join(extensions)}")
        print("\n" + "="*50 + "\n")

        # Run the workflow
        print("Starting agent workflow...")
        result = workflow.run(
            folder_path=str(folder_path),
            file_extensions=extensions
        )

        # Check for errors
        if "error" in result:
            print(f"Error during workflow execution: {result['error']}")
            print(f"Message: {result.get('message', 'No additional message provided.')}")
            return 1

        # Display the result
        print("\nWorkflow completed successfully!")
        print("\nGenerated Persona Information:")
        print(f"Name: {result['persona']['name']}")
        print(f"Writing Voice: {result['persona']['writing_voice_summary']}")
        
        print("\nStyle Markers:")
        for marker in result['persona']['style_markers']:
            print(f"- {marker}")
        
        print("\nRecommended Topics:")
        for topic in result['persona']['recommended_topics']:
            print(f"- {topic}")
        
        print("\nSample Response:")
        print(f"\"{result['sample_response']}\"")
        
        print("\nOutput Files:")
        for file_type, file_path in result['files'].items():
            print(f"- {file_type}: {output_dir / file_path}")
        
        print("\n" + "="*50)
        print("\nWorkflow execution completed.")
        
        # Save the full result to a file
        result_path = output_dir / "workflow_result.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Full result saved to: {result_path}")
        
        return 0

    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
