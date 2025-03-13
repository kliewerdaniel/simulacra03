from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, BackgroundTasks
from fastapi.responses import JSONResponse
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
from simulacra.src.persona_generator.persona_generation_agent import PersonaGenerationAgent
from simulacra.src.persona_generator.persona import Persona

# For dependency injection
from ..auth.auth import get_current_active_user, User

router = APIRouter()

# Temporary file storage - in production use a proper file storage system
UPLOAD_DIR = Path("./uploads")
PERSONA_DIR = Path("./personas")

# Ensure directories exist
UPLOAD_DIR.mkdir(exist_ok=True)
PERSONA_DIR.mkdir(exist_ok=True)

class PersonaTask:
    """Class to track persona generation tasks and their status"""
    def __init__(self, task_id: str, user_id: str):
        self.task_id = task_id
        self.user_id = user_id
        self.status = "pending"
        self.result = None
        self.created_at = datetime.now()
        self.completed_at = None
        self.error = None

# In-memory storage for tasks - in production use a database
persona_tasks = {}
# In-memory storage for personas - in production use a database
personas = {}

def generate_persona_from_analysis_data(task_id: str, analysis_data: Dict[str, Any], api_key: Optional[str] = None):
    """Background task to generate persona from analysis data"""
    try:
        # Update task status
        task = persona_tasks.get(task_id)
        if not task:
            return
        
        task.status = "processing"
        
        # Create persona generation agent
        agent = PersonaGenerationAgent(api_key=api_key)
        
        # Convert analysis data to AuthorAnalysis
        from simulacra.src.document_analysis.document_analyzer import AuthorAnalysis
        analysis = AuthorAnalysis.model_validate(analysis_data)
        
        # Generate persona from analysis
        author_persona = agent.generate_persona_from_analysis(analysis)
        
        # Save persona result
        result_path = PERSONA_DIR / f"{task_id}.json"
        with open(result_path, "w") as f:
            json.dump(author_persona.model_dump(), f, indent=2)
        
        # Store persona in memory
        personas[task_id] = author_persona
        
        # Update task status
        task.status = "completed"
        task.result = str(result_path)
        task.completed_at = datetime.now()
        
    except Exception as e:
        # Update task with error
        if task_id in persona_tasks:
            task = persona_tasks[task_id]
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()

def generate_persona_task(task_id: str, folder_path: str, file_extensions: List[str], api_key: Optional[str] = None):
    """Background task to generate persona from documents"""
    try:
        # Update task status
        task = persona_tasks.get(task_id)
        if not task:
            return
        
        task.status = "processing"
        
        # Create persona generation agent
        agent = PersonaGenerationAgent(api_key=api_key)
        
        # Generate persona from documents
        author_persona = agent.generate_from_documents(folder_path, file_extensions)
        
        # Save persona result
        result_path = PERSONA_DIR / f"{task_id}.json"
        with open(result_path, "w") as f:
            json.dump(author_persona.model_dump(), f, indent=2)
        
        # Store persona in memory
        personas[task_id] = author_persona
        
        # Update task status
        task.status = "completed"
        task.result = str(result_path)
        task.completed_at = datetime.now()
        
    except Exception as e:
        # Update task with error
        if task_id in persona_tasks:
            task = persona_tasks[task_id]
            task.status = "failed"
            task.error = str(e)
            task.completed_at = datetime.now()

@router.post("/generate-from-analysis/{analysis_id}")
async def generate_persona_from_analysis(
    analysis_id: str,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Generate a persona from an existing document analysis
    """
    # Check if analysis exists
    from ..api.document_analysis import analysis_tasks, ANALYSIS_DIR
    
    if analysis_id not in analysis_tasks:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis_task = analysis_tasks[analysis_id]
    
    # Ensure user can only access their own tasks
    if analysis_task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this analysis")
    
    # Check if analysis is completed
    if analysis_task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Analysis is not completed. Current status: {analysis_task.status}")
    
    # Generate a unique ID for this task
    task_id = str(uuid.uuid4())
    
    try:
        # Load the analysis result
        result_path = ANALYSIS_DIR / f"{analysis_id}.json"
        
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Analysis result file not found")
        
        with open(result_path, "r") as f:
            analysis_data = json.load(f)
        
        # Create a task
        task = PersonaTask(task_id=task_id, user_id=current_user.username)
        persona_tasks[task_id] = task
        
        # Start background task to generate persona from analysis
        background_tasks.add_task(
            generate_persona_from_analysis_data,
            task_id,
            analysis_data,
            api_key
        )
        
        return {"task_id": task_id, "status": "pending"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_persona(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    file_extensions: Optional[str] = Form("txt,md,docx,pdf"),
    api_key: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Upload documents and start persona generation process
    """
    # Generate a unique ID for this task
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
        task = PersonaTask(task_id=task_id, user_id=current_user.username)
        persona_tasks[task_id] = task
        
        # Start generation in background
        background_tasks.add_task(
            generate_persona_task, 
            task_id, 
            str(task_dir), 
            extensions_list,
            api_key
        )
        
        return {"task_id": task_id, "status": "pending"}
        
    except Exception as e:
        # Clean up on error
        shutil.rmtree(task_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{task_id}")
async def get_persona_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of a persona generation task"""
    if task_id not in persona_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = persona_tasks[task_id]
    
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
        response["result_url"] = f"/api/persona-generation/result/{task_id}"
    
    if task.status == "failed":
        response["error"] = task.error
    
    return response

@router.get("/result/{task_id}")
async def get_persona_result(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the result of a completed persona generation task"""
    if task_id not in persona_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = persona_tasks[task_id]
    
    # Ensure user can only access their own tasks
    if task.user_id != current_user.username:
        raise HTTPException(status_code=403, detail="Not authorized to access this task")
    
    if task.status != "completed":
        raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    # Load the persona result
    result_path = PERSONA_DIR / f"{task_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        with open(result_path, "r") as f:
            persona_data = json.load(f)
        
        return persona_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result: {str(e)}")

@router.post("/create-simple")
async def create_simple_persona(
    name: str = Form(...),
    traits: List[str] = Form(...),
    background: str = Form(...),
    communication_style: str = Form(...),
    knowledge_areas: Optional[List[str]] = Form(None),
    additional_details: Optional[Dict[str, Any]] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Create a simple persona without document analysis"""
    try:
        # Generate a unique ID for this persona
        persona_id = str(uuid.uuid4())
        
        # Create the persona
        persona = Persona(
            id=persona_id,
            name=name,
            traits=traits,
            background=background,
            communication_style=communication_style,
            knowledge_areas=knowledge_areas or [],
            additional_details=additional_details or {}
        )
        
        # Save persona to file
        result_path = PERSONA_DIR / f"{persona_id}.json"
        with open(result_path, "w") as f:
            json.dump(persona.model_dump(), f, indent=2)
        
        # Store in memory
        personas[persona_id] = persona
        
        return {"persona_id": persona_id, "persona": persona.model_dump()}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating persona: {str(e)}")

@router.get("/list")
async def list_personas(
    current_user: User = Depends(get_current_active_user)
):
    """List all personas available to the user"""
    user_personas = []
    
    # Find all completed persona tasks for this user
    for task_id, task in persona_tasks.items():
        if task.user_id == current_user.username and task.status == "completed":
            result_path = PERSONA_DIR / f"{task_id}.json"
            if result_path.exists():
                try:
                    with open(result_path, "r") as f:
                        persona_data = json.load(f)
                    
                    # Add basic info to the list
                    user_personas.append({
                        "id": task_id,
                        "name": persona_data.get("name", "Unnamed"),
                        "created_at": task.created_at.isoformat(),
                        "type": "generated"
                    })
                except Exception:
                    # Skip invalid files
                    pass
    
    # Also look for manually created personas that are in memory
    for persona_id, persona in personas.items():
        if persona_id not in persona_tasks:  # Skip if it's a generated persona already counted
            user_personas.append({
                "id": persona_id,
                "name": persona.name,
                "created_at": datetime.now().isoformat(),
                "type": "manual"
            })
    
    return {"personas": user_personas}

@router.post("/upload-markdown")
async def upload_markdown_persona(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
):
    """Upload and import a persona from a markdown file"""
    # Generate a unique ID for this persona
    persona_id = str(uuid.uuid4())
    
    try:
        # Check file extension
        if not file.filename.lower().endswith(('.md', '.markdown', '.txt')):
            raise HTTPException(
                status_code=400, 
                detail="Invalid file format. Only markdown (.md, .markdown) and text (.txt) files are supported."
            )
        
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file.filename)
        
        try:
            # Write uploaded content to temp file
            content = await file.read()
            temp_file.write(content)
            temp_file.close()
            
            # Load persona from markdown
            from simulacra.src.file_operations.persona_serializer import PersonaSerializer
            serializer = PersonaSerializer()
            persona = serializer.load_persona(temp_file.name)
            
            # Set a unique ID if not already present
            if not persona.id:
                persona.id = persona_id
            
            # Save persona to storage
            result_path = PERSONA_DIR / f"{persona_id}.json"
            with open(result_path, "w") as f:
                json.dump(persona.to_dict(), f, indent=2)
            
            # Store in memory
            personas[persona_id] = persona
            
            return {
                "success": True, 
                "persona_id": persona_id, 
                "persona": persona.to_dict(),
                "message": f"Successfully imported persona '{persona.name}'"
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
    except Exception as e:
        # Handle errors
        error_msg = str(e)
        if "Error loading persona" in error_msg:
            raise HTTPException(status_code=400, detail=f"Invalid persona format: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail=f"Error uploading persona: {error_msg}")

@router.post("/save/{persona_id}")
async def save_persona(
    persona_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Save a persona permanently"""
    # Check if persona exists
    if persona_id not in persona_tasks and persona_id not in personas:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    # For a persona that was generated through a task
    if persona_id in persona_tasks:
        task = persona_tasks[persona_id]
        
        # Ensure user can only access their own tasks
        if task.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this persona")
        
        # Check if the task is completed
        if task.status != "completed":
            raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
        
        # Persona already saved in our implementation if task is completed
        return {"success": True, "message": "Persona saved successfully"}
    
    # For manually created personas
    elif persona_id in personas:
        # Here you could implement moving from temporary to permanent storage if needed
        # For now, just return success as we're already storing them
        return {"success": True, "message": "Persona saved successfully"}

@router.get("/visualize/{persona_id}")
async def visualize_persona(
    persona_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Generate visualization data for the persona"""
    # First check in tasks
    if persona_id in persona_tasks:
        task = persona_tasks[persona_id]
        
        # Ensure user can only access their own tasks
        if task.user_id != current_user.username:
            raise HTTPException(status_code=403, detail="Not authorized to access this persona")
        
        if task.status != "completed":
            raise HTTPException(status_code=400, detail=f"Task is not completed. Current status: {task.status}")
    
    # Load the persona data
    result_path = PERSONA_DIR / f"{persona_id}.json"
    
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Persona not found")
    
    try:
        with open(result_path, "r") as f:
            persona_data = json.load(f)
        
        # Extract visualization data from the persona
        visualization_data = {}
        
        # If it's an AuthorPersona (has writing_characteristics)
        if "psychological_traits" in persona_data:
            visualization_data = {
                "personality_traits": {
                    "labels": [
                        "Openness", 
                        "Conscientiousness", 
                        "Extraversion", 
                        "Agreeableness", 
                        "Neuroticism"
                    ],
                    "values": [
                        persona_data["psychological_traits"]["openness"],
                        persona_data["psychological_traits"]["conscientiousness"],
                        persona_data["psychological_traits"]["extraversion"],
                        persona_data["psychological_traits"]["agreeableness"],
                        persona_data["psychological_traits"]["neuroticism"]
                    ]
                },
                "writing_traits": {
                    "labels": [
                        "Formality", 
                        "Analytical Thinking", 
                        "Emotional Expression", 
                        "Confidence"
                    ],
                    "values": [
                        persona_data["psychological_traits"]["formality_level"],
                        persona_data["psychological_traits"]["analytical_thinking"],
                        persona_data["psychological_traits"]["emotional_expressiveness"],
                        persona_data["psychological_traits"]["confidence_level"]
                    ]
                }
            }
            
            # Add vocabulary stats if available
            if "writing_characteristics" in persona_data and "vocabulary_stats" in persona_data["writing_characteristics"]:
                visualization_data["vocabulary"] = persona_data["writing_characteristics"]["vocabulary_stats"]
            
        # For simpler Persona objects
        else:
            # Create a simple visualization with the data we have
            visualization_data = {
                "traits": {
                    "values": persona_data.get("traits", []),
                },
                "knowledge_areas": {
                    "values": persona_data.get("knowledge_areas", []),
                }
            }
            
        return visualization_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating visualization: {str(e)}")
