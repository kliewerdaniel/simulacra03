from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
import os
import tempfile
import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path

# Import from simulacra library
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from simulacra.src.document_analysis.document_analyzer import DocumentAnalysisAgent, AuthorAnalysis

# For dependency injection
from ..auth.auth import get_current_active_user, User

router = APIRouter()

# Temporary file storage - in production use a proper file storage system
UPLOAD_DIR = Path("./uploads")
ANALYSIS_DIR = Path("./analyses")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

class AnalysisTask:
    """Class to track analysis tasks and their status"""
    def __init__(self, task_id: str, user_id: str):
        self.task_id = task_id
        self.user_id = user_id
        self.status = "pending"
        self.result = None
        self.created_at = datetime.now()
        self.completed_at = None
        self.error = None

# In-memory storage for tasks - in production use a database
analysis_tasks = {}

def analyze_documents_task(task_id: str, folder_path: str, file_extensions: List[str], api_key: Optional[str] = None):
    """Background task to analyze documents"""
    try:
        # Update task status
        task = analysis_tasks.get(task_id)
        if not task:
            return
        
        task.status = "processing"
        
        # Validate API key
        if not api_key:
            # Try to get from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            
        # If still no API key, fail with clear message
        if not api_key:
            raise ValueError("OpenAI API key is required for document analysis. Please provide an API key in the form or set the OPENAI_API_KEY environment variable.")
        
        # Create analysis agent
        agent = DocumentAnalysisAgent(api_key=api_key)
        
        # Analyze documents
        print(f"Starting document analysis for task {task_id}...")
        analysis = agent.analyze_documents(folder_path, file_extensions)
        
        # Save analysis result
        result_path = ANALYSIS_DIR / f"{task_id}.json"
        with open(result_path, "w") as f:
            json.dump(analysis.model_dump(), f, indent=2)
        
        # Generate readable report
        report_path = ANALYSIS_DIR / f"{task_id}_report.md"
        agent.generate_analysis_report(analysis, str(report_path))
        
        # Update task status
        task.status = "completed"
        task.result = str(result_path)
        task.completed_at = datetime.now()
        print(f"Document analysis completed for task {task_id}")
        
    except ValueError as e:
        # Handle validation errors (like missing API key)
        if task_id in analysis_tasks:
            task = analysis_tasks[task_id]
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()
            print(f"Document analysis failed for task {task_id}: {str(e)}")
    except Exception as e:
        # Handle other errors
        if task_id in analysis_tasks:
            task = analysis_tasks[task_id]
            task.status = "failed"
            error_message = f"Error during document analysis: {str(e)}"
            task.error = error_message
            task.completed_at = datetime.now()
            print(f"Document analysis failed for task {task_id}: {error_message}")
            import traceback
            traceback.print_exc()

@router.post("/analyze")
async def analyze_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    file_extensions: Optional[str] = Form("txt,md,docx,pdf"),
    api_key: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload documents and start analysis process
    """
    # Check if any files were uploaded
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")
    
    # Generate a unique ID for this analysis task
    task_id = str(uuid.uuid4())
    
    # Create a directory for this task's files
    task_dir = UPLOAD_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    
    try:
        # Save uploaded files
        for file in files:
            file_path = task_dir / file.filename
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        
        # Parse file extensions
        extensions_list = [ext.strip() for ext in file_extensions.split(",")]
        
        # Create task
        task = AnalysisTask(task_id=task_id, user_id=current_user.username)
        analysis_tasks[task_id] = task
        
        # Validate API key presence
        api_key_message = ""
        if not api_key and not os.environ.get("OPENAI_API_KEY"):
            api_key_message = (" An OpenAI API key is required for document analysis. "
                               "Either provide one in the form or set the OPENAI_API_KEY environment variable.")
        
        # Start analysis in background
        background_tasks.add_task(
            analyze_documents_task, 
            task_id, 
            str(task_dir), 
            extensions_list,
            api_key
        )
        
        response = {
            "task_id": task_id, 
            "status": "pending", 
            "files_uploaded": len(files),
            "message": f"Analysis started.{api_key_message}"
        }
        
        return response
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_analysis_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of an analysis task"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    # Ensure user can only access their own tasks
    if task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    response = {
        "task_id": task.task_id,
        "status": task.status,
        "created_at": task.created_at.isoformat(),
    }
    
    if task.completed_at:
        response["completed_at"] = task.completed_at.isoformat()
    
    if task.status == "completed":
        response["result_url"] = f"/api/document-analysis/result/{task_id}"
    
    if task.status == "failed":
        response["error"] = task.error
        # If error is about API key, add a more helpful message
        if "API key" in task.error:
            response["help"] = "Please provide an OpenAI API key in the form or set it in your environment variables."
    
    return response

@router.get("/result/{task_id}")
async def get_analysis_result(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the result of a completed analysis task"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    # Ensure user can only access their own tasks
    if task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    # Load the analysis result
    result_path = ANALYSIS_DIR / f"{task_id}.json"
    report_path = ANALYSIS_DIR / f"{task_id}_report.md"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(result_path, "r") as f:
            analysis_data = json.load(f)
        
        report_content = ""
        if report_path.exists():
            with open(report_path, "r") as f:
                report_content = f.read()
        
        return {
            "analysis": analysis_data,
            "report": report_content
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

@router.get("/result/{task_id}/download")
async def download_analysis_report(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Download the analysis report as a file"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    # Ensure user can only access their own tasks
    if task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    # Get the report file path
    report_path = ANALYSIS_DIR / f"{task_id}_report.md"
    
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="Report file not found")
    
    # Return the file as a downloadable response
    return FileResponse(
        path=report_path,
        filename=f"document_analysis_report_{task_id}.md",
        media_type="text/markdown"
    )

@router.get("/visualize/{task_id}")
async def visualize_analysis(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Generate visualization data for the analysis"""
    if task_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = analysis_tasks[task_id]
    
    # Ensure user can only access their own tasks
    if task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    # Load the analysis result
    result_path = ANALYSIS_DIR / f"{task_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(result_path, "r") as f:
            analysis_data = json.load(f)
        
        # Extract visualization data from the analysis
        visualization_data = {
            "personality_radar": {
                "labels": [
                    "Openness", 
                    "Conscientiousness", 
                    "Extraversion", 
                    "Agreeableness", 
                    "Neuroticism"
                ],
                "values": [
                    analysis_data["psychological_profile"]["openness"],
                    analysis_data["psychological_profile"]["conscientiousness"],
                    analysis_data["psychological_profile"]["extraversion"],
                    analysis_data["psychological_profile"]["agreeableness"],
                    analysis_data["psychological_profile"]["neuroticism"]
                ]
            },
            "writing_style": {
                "labels": [
                    "Formality", 
                    "Analytical Thinking", 
                    "Emotional Expression", 
                    "Confidence"
                ],
                "values": [
                    analysis_data["psychological_profile"]["formality_level"],
                    analysis_data["psychological_profile"]["analytical_thinking"],
                    analysis_data["psychological_profile"]["emotional_expressiveness"],
                    analysis_data["psychological_profile"]["confidence_level"]
                ]
            },
            "word_frequencies": {
                "labels": list(analysis_data["features"]["word_frequencies"].keys())[:20],
                "values": list(analysis_data["features"]["word_frequencies"].values())[:20]
            },
            "sentence_structures": {
                "labels": list(analysis_data["features"]["sentence_structures"].keys()),
                "values": list(analysis_data["features"]["sentence_structures"].values())
            }
        }
        
        return visualization_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")
