import os
from fastapi import FastAPI, Depends, HTTPException, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pathlib import Path
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

from .api import document_analysis, persona_generation, style_replication, ecommerce, content_management
from .auth import auth

# Create the FastAPI app
app = FastAPI(
    title="Simulacra Web API",
    description="Web API for the Simulacra document analysis and persona generation tool",
    version="0.1.0",
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="app/templates")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(
    document_analysis.router, 
    prefix="/api/document-analysis", 
    tags=["document-analysis"],
    dependencies=[Depends(auth.get_current_active_user)]
)
app.include_router(
    persona_generation.router, 
    prefix="/api/persona-generation", 
    tags=["persona-generation"],
    dependencies=[Depends(auth.get_current_active_user)]
)
app.include_router(
    style_replication.router, 
    prefix="/api/style-replication", 
    tags=["style-replication"],
    dependencies=[Depends(auth.get_current_active_user)]
)
app.include_router(
    ecommerce.router, 
    prefix="/api/ecommerce", 
    tags=["ecommerce"]
)
app.include_router(
    content_management.router, 
    prefix="/api/content-management", 
    tags=["content-management"],
    dependencies=[Depends(auth.get_current_active_user)]
)

# Root route - redirect to the web UI
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    # Provide user=None to avoid the request.user error
    context = {"request": request, "user": None}
    return templates.TemplateResponse("index.html", context)

# Dashboard route - handling both GET and POST
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_get(request: Request):
    # First try to get token from Authorization header
    auth_header = request.headers.get("Authorization")
    token = None
    
    # Extract token if Authorization header exists
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    # If no auth header, try to get from cookie or query param
    if not token:
        token = request.cookies.get("auth_token")
    
    if not token:
        # No valid token found, redirect to login
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify the token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        # Return dashboard with user
        return templates.TemplateResponse("dashboard.html", {"request": request, "user": user})
    
    except auth.JWTError:
        # Invalid token, redirect to login
        return RedirectResponse(url="/auth/login", status_code=303)

@app.post("/dashboard", response_class=HTMLResponse)
async def dashboard_post(request: Request):
    # Handle POST request to dashboard (from form submission)
    form_data = await request.form()
    token = form_data.get("token")
    
    if not token:
        # Try to get from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
    
    if not token:
        # No valid token found, redirect to login
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify the token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        # Set token as cookie for future requests
        response = templates.TemplateResponse("dashboard.html", {"request": request, "user": user})
        response.set_cookie(key="auth_token", value=token, httponly=True)
        return response
    
    except auth.JWTError:
        # Invalid token, redirect to login
        return RedirectResponse(url="/auth/login", status_code=303)

# Add routes for each feature
@app.get("/document-analysis", response_class=HTMLResponse)
async def document_analysis_page(request: Request):
    # Similar auth check as dashboard route
    auth_header = request.headers.get("Authorization")
    token = None
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    if not token:
        token = request.cookies.get("auth_token")
    
    if not token:
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        return templates.TemplateResponse("document_analysis.html", {"request": request, "user": user})
    
    except auth.JWTError:
        return RedirectResponse(url="/auth/login", status_code=303)

@app.get("/persona-generation", response_class=HTMLResponse)
async def persona_generation_page(request: Request):
    # Similar auth check as dashboard route
    auth_header = request.headers.get("Authorization")
    token = None
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    if not token:
        token = request.cookies.get("auth_token")
    
    if not token:
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        return templates.TemplateResponse("persona_generation.html", {"request": request, "user": user})
    
    except auth.JWTError:
        return RedirectResponse(url="/auth/login", status_code=303)

@app.get("/style-replication", response_class=HTMLResponse)
async def style_replication_page(request: Request):
    # Similar auth check as dashboard route
    auth_header = request.headers.get("Authorization")
    token = None
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    if not token:
        token = request.cookies.get("auth_token")
    
    if not token:
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        return templates.TemplateResponse("style_replication.html", {"request": request, "user": user})
    
    except auth.JWTError:
        return RedirectResponse(url="/auth/login", status_code=303)

@app.get("/content-management", response_class=HTMLResponse)
async def content_management_page(request: Request):
    # Similar auth check as dashboard route
    auth_header = request.headers.get("Authorization")
    token = None
    
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
    
    if not token:
        token = request.cookies.get("auth_token")
    
    if not token:
        return RedirectResponse(url="/auth/login", status_code=303)
    
    try:
        # Verify token and get user
        payload = auth.jwt.decode(token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        user = auth.get_user(auth.fake_users_db, username=username)
        if user is None:
            return RedirectResponse(url="/auth/login", status_code=303)
        
        return templates.TemplateResponse("content_management.html", {"request": request, "user": user})
    
    except auth.JWTError:
        return RedirectResponse(url="/auth/login", status_code=303)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
