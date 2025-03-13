#!/usr/bin/env python3
"""
Run script for the Simulacra Web ecommerce integration.
This starts the FastAPI server on port 8001.
"""

import uvicorn

if __name__ == "__main__":
    print("Starting Simulacra Web Server with ecommerce integration...")
    print("The server will run on http://127.0.0.1:8001")
    print("API documentation is available at http://127.0.0.1:8001/docs")
    print("Press Ctrl+C to stop the server")
    
    uvicorn.run("app.main:app", host="127.0.0.1", port=8001, reload=True)
