#!/usr/bin/env python3
"""
Startup script for Simulacra Web Interface
"""
import os
import sys
import uvicorn

def main():
    # Add parent directory to Python path to access simulacra module
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    
    # Create required directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("analyses", exist_ok=True)
    os.makedirs("personas", exist_ok=True)
    os.makedirs("generated_content", exist_ok=True)
    
    print("Starting Simulacra Web Interface...")
    print("You can access the web interface at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    main()
