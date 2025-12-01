from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import uuid
import tempfile
from .llm import (
    load_from_type, build_qa_chain, generate_summary, generate_preview,
    search_web_with_google, answer_with_gemini, is_answer_not_found
)

app = FastAPI(
    title="Document QA API",
    description="API for document question answering with web search fallback",
    version="1.0.0"
)

# CORS middleware - configured for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage (use Redis in production)
qa_chains: Dict[str, Any] = {}
document_summaries: Dict[str, str] = {}
session_documents: Dict[str, list] = {}

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}


# Pydantic models for request validation
class QueryInput(BaseModel):
    session_id: str
    question: str


class SearchInput(BaseModel):
    question: str


class ContentInput(BaseModel):
    """For URL or text input"""
    url: Optional[str] = None
    text: Optional[str] = None


class AddContextInput(BaseModel):
    """For adding context via URL or text"""
    session_id: str
    url: Optional[str] = None
    text: Optional[str] = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Document QA API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Document QA API"
    }


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process a file (PDF or image)
    Returns session_id, summary, and content_type
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        session_id = str(uuid.uuid4())
        
        # Save file temporarily
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
        
        # Write file content
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Determine file type
        input_type = 'pdf' if file_ext == 'pdf' else 'image'
        
        # Process the file
        docs = load_from_type(input_type, filepath)
        qa_chain = build_qa_chain(docs)
        summary = generate_summary(docs, input_type)
        
        # Store in memory
        qa_chains[session_id] = qa_chain
        document_summaries[session_id] = summary
        session_documents[session_id] = docs
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return {
            'success': True,
            'session_id': session_id,
            'summary': summary,
            'content_type': input_type
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/api/upload-json")
async def upload_content(content: ContentInput):
    """
    Upload and process content via JSON (URL or text)
    Returns session_id, summary, and content_type
    """
    try:
        if not content.url and not content.text:
            raise HTTPException(
                status_code=400, 
                detail="Either 'url' or 'text' must be provided"
            )
        
        session_id = str(uuid.uuid4())
        
        if content.url:
            # Process URL
            docs = load_from_type('url', content.url)
            input_type = 'url'
        else:
            # Process text
            docs = load_from_type('text', content.text)
            input_type = 'text'
        
        qa_chain = build_qa_chain(docs)
        summary = generate_summary(docs, input_type)
        
        # Store in memory
        qa_chains[session_id] = qa_chain
        document_summaries[session_id] = summary
        session_documents[session_id] = docs
        
        return {
            'success': True,
            'session_id': session_id,
            'summary': summary,
            'content_type': input_type
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing content: {str(e)}")


@app.post("/api/query")
async def query_document(query_input: QueryInput):
    """
    Query the uploaded document
    Returns answer and source information
    """
    try:
        if not query_input.session_id:
            raise HTTPException(status_code=400, detail="session_id is required")
        
        if not query_input.question:
            raise HTTPException(status_code=400, detail="question is required")
        
        if query_input.session_id not in qa_chains:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload content first."
            )
        
        qa_chain = qa_chains[query_input.session_id]
        result = qa_chain.invoke({"query": query_input.question})
        response_text = result["result"]
        
        response = {
            'success': True,
            'answer': response_text,
            'source': 'document'
        }
        
        # Check if answer was not found
        if is_answer_not_found(response_text):
            response['not_found'] = True
            response['can_search_web'] = True
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")


@app.post("/api/search-web")
async def search_web(search_input: SearchInput):
    """
    Search the web for an answer
    Returns answer from Google Search or Gemini fallback
    """
    try:
        if not search_input.question:
            raise HTTPException(status_code=400, detail="question is required")
        
        # Try Google Search first
        google_result = search_web_with_google(search_input.question)
        
        if google_result == "quota_exceeded":
            # Fallback to Gemini
            gemini_result = answer_with_gemini(search_input.question)
            return {
                'success': True,
                'answer': gemini_result,
                'source': 'gemini_fallback',
                'message': 'Google Search quota exceeded. Used Gemini\'s general knowledge.'
            }
        elif google_result.startswith("search_error:"):
            # Fallback to Gemini
            gemini_result = answer_with_gemini(search_input.question)
            return {
                'success': True,
                'answer': gemini_result,
                'source': 'gemini_fallback',
                'message': 'Google search error. Used Gemini\'s general knowledge.'
            }
        else:
            return {
                'success': True,
                'answer': google_result,
                'source': 'google_search'
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching web: {str(e)}")


@app.post("/api/add-context")
async def add_context_file(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Add additional context to existing session via file upload
    Returns updated summary and added content info
    """
    try:
        if not session_id or session_id not in session_documents:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload initial content first."
            )
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Save file temporarily
        file_ext = file.filename.rsplit('.', 1)[1].lower()
        filepath = os.path.join(UPLOAD_FOLDER, f"{session_id}_{file.filename}")
        
        with open(filepath, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Determine file type
        input_type = 'pdf' if file_ext == 'pdf' else 'image'
        
        # Load new documents
        new_docs = load_from_type(input_type, filepath)
        
        # Generate preview of new content
        new_content_preview = generate_preview(new_docs, f"{input_type} file: {file.filename}")
        
        # Add to existing documents
        session_documents[session_id].extend(new_docs)
        
        # Rebuild QA chain with all documents
        qa_chain = build_qa_chain(session_documents[session_id])
        summary = generate_summary(session_documents[session_id], 'mixed')
        
        # Update stored data
        qa_chains[session_id] = qa_chain
        document_summaries[session_id] = summary
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return {
            'success': True,
            'summary': summary,
            'added_content': {
                'type': input_type,
                'name': file.filename,
                'preview': new_content_preview
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding context: {str(e)}")


@app.post("/api/add-context-json")
async def add_context_json(context: AddContextInput):
    """
    Add additional context to existing session via JSON (URL or text)
    Returns updated summary and added content info
    """
    try:
        if not context.session_id or context.session_id not in session_documents:
            raise HTTPException(
                status_code=404, 
                detail="Session not found. Please upload initial content first."
            )
        
        if not context.url and not context.text:
            raise HTTPException(
                status_code=400, 
                detail="Either 'url' or 'text' must be provided"
            )
        
        if context.url:
            # Add URL context
            new_docs = load_from_type('url', context.url)
            new_content_preview = generate_preview(new_docs, f"URL: {context.url}")
            content_type = 'url'
            content_name = context.url
        else:
            # Add text context
            new_docs = load_from_type('text', context.text)
            new_content_preview = generate_preview(new_docs, "Text content")
            content_type = 'text'
            content_name = 'Custom text'
        
        # Add to existing documents
        session_documents[context.session_id].extend(new_docs)
        
        # Rebuild QA chain with all documents
        qa_chain = build_qa_chain(session_documents[context.session_id])
        summary = generate_summary(session_documents[context.session_id], 'mixed')
        
        # Update stored data
        qa_chains[context.session_id] = qa_chain
        document_summaries[context.session_id] = summary
        
        return {
            'success': True,
            'summary': summary,
            'added_content': {
                'type': content_type,
                'name': content_name,
                'preview': new_content_preview
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding context: {str(e)}")


@app.get("/api/sessions/{session_id}/summary")
async def get_summary(session_id: str):
    """
    Get summary for a specific session
    Returns the document summary
    """
    if session_id not in document_summaries:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        'success': True,
        'summary': document_summaries[session_id]
    }


@app.get("/api/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    """
    Get detailed information about a session
    Returns session metadata and statistics
    """
    if session_id not in session_documents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    docs = session_documents[session_id]
    total_chars = sum(len(doc.page_content) for doc in docs)
    
    return {
        'success': True,
        'session_id': session_id,
        'document_count': len(docs),
        'total_characters': total_chars,
        'summary': document_summaries.get(session_id, "No summary available"),
        'has_qa_chain': session_id in qa_chains
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and free up memory
    Returns success status
    """
    if session_id not in session_documents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Clean up session data
    if session_id in qa_chains:
        del qa_chains[session_id]
    if session_id in document_summaries:
        del document_summaries[session_id]
    if session_id in session_documents:
        del session_documents[session_id]
    
    return {
        'success': True,
        'message': f'Session {session_id} deleted successfully'
    }


@app.get("/api/sessions")
async def list_sessions():
    """
    List all active sessions
    Returns list of session IDs with basic info
    """
    sessions = []
    for session_id in session_documents.keys():
        sessions.append({
            'session_id': session_id,
            'document_count': len(session_documents[session_id]),
            'has_summary': session_id in document_summaries
        })
    
    return {
        'success': True,
        'count': len(sessions),
        'sessions': sessions
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)