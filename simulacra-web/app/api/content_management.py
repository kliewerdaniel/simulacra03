from fastapi import APIRouter, Depends, HTTPException, Form, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Optional, Dict, Any
import os
import json
import uuid
import shutil
from datetime import datetime
from pathlib import Path

# Import from simulacra library
import sys
import os

# For dependency injection
from ..auth.auth import get_current_active_user, User

# Import paths from other modules
from .persona_generation import PERSONA_DIR
from .style_replication import CONTENT_DIR, generation_tasks
from .document_analysis import ANALYSIS_DIR

router = APIRouter()

# Create directories if they don't exist
PERSONA_DIR.mkdir(exist_ok=True)
CONTENT_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

@router.get("/list-all")
async def list_all_content(
    current_user: User = Depends(get_current_active_user)
):
    """List all types of content: personas, generated content, and analysis reports"""
    
    # Get personas
    personas = []
    if PERSONA_DIR.exists():
        for file_path in PERSONA_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    personas.append({
                        "id": file_path.stem,
                        "name": data.get("name", "Unnamed Persona"),
                        "type": "persona",
                        "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "description": data.get("description", "No description"),
                        "file_path": str(file_path)
                    })
            except Exception as e:
                # Skip invalid files
                pass
    
    # Get generated content 
    content_items = []
    if CONTENT_DIR.exists():
        for file_path in CONTENT_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Extract brief info, handling different structures
                    brief = data.get("content_brief", {})
                    if isinstance(brief, dict):
                        topic = brief.get("topic", "Unknown topic")
                        content_type = brief.get("content_type", "Unknown type")
                    else:
                        topic = "Unknown topic"
                        content_type = "Unknown type"
                    
                    content_items.append({
                        "id": file_path.stem,
                        "topic": topic,
                        "content_type": content_type,
                        "type": "generated_content",
                        "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "preview": data.get("content", "")[:100] + "..." if len(data.get("content", "")) > 100 else data.get("content", ""),
                        "file_path": str(file_path)
                    })
            except Exception as e:
                # Skip invalid files
                pass
    
    # Get analysis reports
    analyses = []
    if ANALYSIS_DIR.exists():
        for file_path in ANALYSIS_DIR.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    
                    # Extract key info
                    analyses.append({
                        "id": file_path.stem,
                        "document_id": data.get("document_id", "Unknown"),
                        "type": "analysis",
                        "created_at": datetime.fromtimestamp(file_path.stat().st_ctime).isoformat(),
                        "summary": data.get("summary", "No summary available")[:100] + "..." if len(data.get("summary", "")) > 100 else data.get("summary", ""),
                        "file_path": str(file_path)
                    })
            except Exception as e:
                # Skip invalid files
                pass
                
    return {
        "personas": personas,
        "content": content_items,
        "analyses": analyses
    }

@router.delete("/delete/{content_type}/{content_id}")
async def delete_content(
    content_type: str,
    content_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete content by type and ID"""
    
    # Determine the directory based on content type
    if content_type == "persona":
        directory = PERSONA_DIR
    elif content_type == "generated_content":
        directory = CONTENT_DIR
    elif content_type == "analysis":
        directory = ANALYSIS_DIR
    else:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
    
    # Check if file exists
    file_path = directory / f"{content_id}.json"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{content_type} with ID {content_id} not found")
    
    # Delete the file
    try:
        os.remove(file_path)
        
        # For generated content, also delete the .txt file if it exists
        if content_type == "generated_content":
            txt_path = directory / f"{content_id}.txt"
            if txt_path.exists():
                os.remove(txt_path)
            
            # Remove from in-memory store if it exists
            if content_id in generation_tasks:
                del generation_tasks[content_id]
        
        # For analysis, also delete the markdown report if it exists
        if content_type == "analysis":
            md_path = directory / f"{content_id}_report.md"
            if md_path.exists():
                os.remove(md_path)
        
        return {"status": "success", "message": f"{content_type} with ID {content_id} deleted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting {content_type}: {str(e)}")

@router.post("/import/{content_type}")
async def import_content(
    content_type: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Import content from a JSON file"""
    
    # Validate file extension
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")
    
    # Determine the directory based on content type
    if content_type == "persona":
        directory = PERSONA_DIR
    elif content_type == "generated_content":
        directory = CONTENT_DIR
    elif content_type == "analysis":
        directory = ANALYSIS_DIR
    else:
        raise HTTPException(status_code=400, detail=f"Invalid content type: {content_type}")
    
    # Generate a unique ID
    content_id = str(uuid.uuid4())
    file_path = directory / f"{content_id}.json"
    
    try:
        # Read and validate the uploaded file
        content = await file.read()
        json_content = json.loads(content.decode())
        
        # Basic validation based on content type
        if content_type == "persona" and "name" not in json_content:
            raise HTTPException(status_code=400, detail="Invalid persona format: missing 'name' field")
        elif content_type == "generated_content" and "content" not in json_content:
            raise HTTPException(status_code=400, detail="Invalid content format: missing 'content' field")
        elif content_type == "analysis" and "document_id" not in json_content:
            raise HTTPException(status_code=400, detail="Invalid analysis format: missing 'document_id' field")
        
        # Save the file
        with open(file_path, "wb") as f:
            f.write(content)
        
        # For generated content, also create a .txt file
        if content_type == "generated_content" and "content" in json_content:
            txt_path = directory / f"{content_id}.txt"
            with open(txt_path, "w") as f:
                f.write(json_content["content"])
        
        return {
            "status": "success", 
            "id": content_id, 
            "message": f"{content_type} imported successfully"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error importing {content_type}: {str(e)}")
