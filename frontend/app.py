import streamlit as st
import requests
from typing import Optional
import os
from io import BytesIO

# Configuration
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="RAG Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid;
        color: #1f1f1f;
    }
    .user-message {
        background-color: #f0f2f6;
        border-left-color: #1f77b4;
        color: #1f1f1f;
    }
    .assistant-message {
        background-color: #e8f4f8;
        border-left-color: #2ecc71;
        color: #1f1f1f;
    }
    .web-search-message {
        background-color: #fff3cd;
        border-left-color: #ffc107;
        color: #1f1f1f;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_summary' not in st.session_state:
    st.session_state.document_summary = None
if 'content_type' not in st.session_state:
    st.session_state.content_type = None
if 'context_added' not in st.session_state:
    st.session_state.context_added = []


def upload_file(file):
    """Upload a file to the API"""
    try:
        files = {'file': (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/api/upload", files=files)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error uploading file: {str(e)}")
        return None


def upload_url(url: str):
    """Upload a URL to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/upload-json",
            json={'url': url}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error processing URL: {str(e)}")
        return None


def upload_text(text: str):
    """Upload text to the API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/upload-json",
            json={'text': text}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return None


def query_document(session_id: str, question: str):
    """Query the uploaded document"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={'session_id': session_id, 'question': question}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying document: {str(e)}")
        return None


def search_web(question: str):
    """Search the web for an answer"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/search-web",
            json={'question': question}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error searching web: {str(e)}")
        return None


def add_context_file(session_id: str, file):
    """Add additional context via file"""
    try:
        files = {'file': (file.name, file, file.type)}
        data = {'session_id': session_id}
        response = requests.post(
            f"{API_BASE_URL}/api/add-context",
            files=files,
            data=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error adding context: {str(e)}")
        return None


def add_context_url(session_id: str, url: str):
    """Add additional context via URL"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/add-context-json",
            json={'session_id': session_id, 'url': url}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error adding context: {str(e)}")
        return None


def add_context_text(session_id: str, text: str):
    """Add additional context via text"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/add-context-json",
            json={'session_id': session_id, 'text': text}
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error adding context: {str(e)}")
        return None


def display_chat_message(role: str, content: str, source: str = "document"):
    """Display a chat message with styling"""
    if role == "user":
        st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong><br>{content}
            </div>
        """, unsafe_allow_html=True)
    else:
        icon = "🤖" if source == "document" else "🌐" if source == "google_search" else "💡"
        css_class = "assistant-message" if source == "document" else "web-search-message"
        source_label = "Document" if source == "document" else "Web Search" if source == "google_search" else "AI Assistant"
        
        st.markdown(f"""
            <div class="chat-message {css_class}">
                <strong>{icon} {source_label}:</strong><br>{content}
            </div>
        """, unsafe_allow_html=True)


# Main app layout
st.title("📚 RAG Agent")
st.markdown("Upload documents, paste URLs, or enter text to ask questions about your content.")

# Sidebar for document upload
with st.sidebar:
    st.header("📤 Upload Content")
    
    upload_method = st.radio(
        "Choose input method:",
        ["File Upload", "URL", "Text Input"],
        key="upload_method"
    )
    
    uploaded_content = None
    
    if upload_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
            help="Upload a PDF document or image"
        )
        
        if uploaded_file and st.button("Process File", type="primary"):
            with st.spinner("Processing file..."):
                result = upload_file(uploaded_file)
                if result and result.get('success'):
                    st.session_state.session_id = result['session_id']
                    st.session_state.document_summary = result['summary']
                    st.session_state.content_type = result['content_type']
                    st.session_state.chat_history = []
                    st.session_state.context_added = []
                    st.success("✅ File processed successfully!")
                    st.rerun()
    
    elif upload_method == "URL":
        url_input = st.text_input(
            "Enter URL",
            placeholder="https://example.com/article",
            help="Enter a webpage URL to extract content"
        )
        
        if url_input and st.button("Process URL", type="primary"):
            with st.spinner("Fetching and processing URL..."):
                result = upload_url(url_input)
                if result and result.get('success'):
                    st.session_state.session_id = result['session_id']
                    st.session_state.document_summary = result['summary']
                    st.session_state.content_type = result['content_type']
                    st.session_state.chat_history = []
                    st.session_state.context_added = []
                    st.success("✅ URL processed successfully!")
                    st.rerun()
    
    else:  # Text Input
        text_input = st.text_area(
            "Enter your text",
            height=200,
            placeholder="Paste your text content here...",
            help="Enter any text content you want to query"
        )
        
        if text_input and st.button("Process Text", type="primary"):
            with st.spinner("Processing text..."):
                result = upload_text(text_input)
                if result and result.get('success'):
                    st.session_state.session_id = result['session_id']
                    st.session_state.document_summary = result['summary']
                    st.session_state.content_type = result['content_type']
                    st.session_state.chat_history = []
                    st.session_state.context_added = []
                    st.success("✅ Text processed successfully!")
                    st.rerun()
    
    # Add context section
    if st.session_state.session_id:
        st.divider()
        st.header("➕ Add More Context")
        
        with st.expander("Add additional content", expanded=False):
            context_method = st.selectbox(
                "Add context via:",
                ["File", "URL", "Text"]
            )
            
            if context_method == "File":
                context_file = st.file_uploader(
                    "Add another file",
                    type=['pdf', 'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
                    key="context_file"
                )
                if context_file and st.button("Add File Context"):
                    with st.spinner("Adding context..."):
                        result = add_context_file(st.session_state.session_id, context_file)
                        if result and result.get('success'):
                            st.session_state.document_summary = result['summary']
                            st.session_state.context_added.append(result['added_content'])
                            st.success("✅ Context added!")
                            st.rerun()
            
            elif context_method == "URL":
                context_url = st.text_input("Add URL context", key="context_url")
                if context_url and st.button("Add URL Context"):
                    with st.spinner("Adding context..."):
                        result = add_context_url(st.session_state.session_id, context_url)
                        if result and result.get('success'):
                            st.session_state.document_summary = result['summary']
                            st.session_state.context_added.append(result['added_content'])
                            st.success("✅ Context added!")
                            st.rerun()
            
            else:  # Text
                context_text = st.text_area("Add text context", key="context_text", height=150)
                if context_text and st.button("Add Text Context"):
                    with st.spinner("Adding context..."):
                        result = add_context_text(st.session_state.session_id, context_text)
                        if result and result.get('success'):
                            st.session_state.document_summary = result['summary']
                            st.session_state.context_added.append(result['added_content'])
                            st.success("✅ Context added!")
                            st.rerun()
        
        # Show added context
        if st.session_state.context_added:
            st.subheader("📎 Added Context")
            for i, context in enumerate(st.session_state.context_added, 1):
                with st.expander(f"{i}. {context['name']}", expanded=False):
                    st.write(f"**Type:** {context['type']}")
                    if 'preview' in context:
                        st.write(f"**Preview:** {context['preview'].get('text_preview', 'N/A')}")
        
        # Reset button
        st.divider()
        if st.button("🔄 Start New Session", type="secondary"):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.document_summary = None
            st.session_state.content_type = None
            st.session_state.context_added = []
            st.rerun()

# Main content area
if st.session_state.session_id:
    # Display document summary
    with st.expander("📋 Document Summary", expanded=True):
        st.info(st.session_state.document_summary)
        st.caption(f"Content Type: {st.session_state.content_type}")
        if st.session_state.context_added:
            st.caption(f"Additional context sources: {len(st.session_state.context_added)}")
    
    # Display chat history
    st.subheader("💬 Conversation")
    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            display_chat_message(
                message['role'],
                message['content'],
                message.get('source', 'document')
            )
    
    # Question input
    st.divider()
    question = st.text_input(
        "Ask a question about your document:",
        placeholder="What is the main topic discussed?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🚀 Ask", type="primary", use_container_width=True)
    with col2:
        search_web_button = st.button("🌐 Search Web Instead", use_container_width=True)
    
    if ask_button and question:
        # Add user message to chat
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Query the document
        with st.spinner("Thinking..."):
            result = query_document(st.session_state.session_id, question)
            
            if result and result.get('success'):
                answer = result['answer']
                source = result.get('source', 'document')
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'source': source
                })
                
                # Check if answer was not found
                if result.get('not_found'):
                    st.warning("⚠️ The answer wasn't found in your document. Would you like to search the web?")
        
        st.rerun()
    
    if search_web_button and question:
        # Add user message to chat
        st.session_state.chat_history.append({
            'role': 'user',
            'content': question
        })
        
        # Search the web
        with st.spinner("Searching the web..."):
            result = search_web(question)
            
            if result and result.get('success'):
                answer = result['answer']
                source = result.get('source', 'google_search')
                
                # Add assistant response to chat
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': answer,
                    'source': source
                })
                
                if result.get('message'):
                    st.info(result['message'])
        
        st.rerun()

else:
    # Welcome screen
    st.info("👈 Please upload a document, paste a URL, or enter text to get started!")
    
    st.markdown("""
    ### How to use:
    
    1. **Choose your input method** in the sidebar:
       - 📄 Upload a PDF or image file
       - 🔗 Paste a URL to extract content
       - ✍️ Enter text directly
    
    2. **Process your content** by clicking the appropriate button
    
    3. **Ask questions** about your content in the chat interface
    
    4. **Add more context** if needed to enrich your knowledge base
    
    5. **Search the web** if the answer isn't in your documents
    
    ### Features:
    - 🤖 AI-powered question answering
    - 📚 Support for multiple document types
    - 🔍 Web search fallback
    - ➕ Add multiple sources to one session
    - 💬 Interactive chat interface
    """)

# Footer
st.divider()
st.caption("Built with ❤️ Ayush Aggarwal")