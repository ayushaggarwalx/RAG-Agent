# 🤖 RAG Agent

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ayushaggarwal-rag-agent.hf.space/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-green)](https://github.com/langchain-ai/langchain)

An intelligent document Q&A assistant powered by LangChain and Google Gemini. Upload PDFs, images, or URLs to chat with your content. Automatically searches the web when answers aren't found in your documents.

## ✨ Features

- 📄 **Multiple Input Formats**: Support for PDF, images (PNG, JPG, etc.), URLs, and plain text
- 🤖 **AI-Powered Q&A**: Uses Google Gemini with LangChain for intelligent question answering
- 🔍 **Web Search Fallback**: Automatically searches the web when answers aren't in your documents
- ➕ **Context Addition**: Add multiple sources to build a comprehensive knowledge base
- 💬 **Interactive Chat**: Beautiful Streamlit interface with conversation history
- 🎨 **Modern UI**: Clean, intuitive design with color-coded responses
- ⚡ **Fast & Efficient**: FAISS vector store for quick similarity search

## 🚀 Try It Now

**Live Demo:** [Try RAG Agent on Hugging Face Spaces](https://ayushaggarwal-rag-agent.hf.space/)

No installation required! Just click the link above to start using RAG Agent immediately.

## 📸 Screenshots

![Upload Interface](UI.png)


## 🏗️ Architecture

```
┌─────────────────────┐
│  Streamlit Frontend │
│   (Port 8501)       │
└──────────┬──────────┘
           │ REST API
           │
┌──────────▼──────────┐
│   FastAPI Backend   │
│   (Port 8000)       │
└──────────┬──────────┘
           │
    ┌──────┴──────┐
    │             │
┌───▼────┐   ┌───▼────┐
│LangChain│   │ Gemini │
│  FAISS  │   │   API  │
└─────────┘   └────────┘
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **LLM Framework**: LangChain
- **AI Model**: Google Gemini (gemini-1.5-flash)
- **Vector Store**: FAISS
- **Embeddings**: Google Generative AI Embeddings
- **Document Loaders**: PyMuPDF, WebBaseLoader, Trafilatura
- **Web Search**: Google Search (via Gemini)

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- Google Gemini API key ([Get it here](https://makersuite.google.com/app/apikey))

### Clone the Repository

```bash
git clone https://github.com/ayushaggarwalx/rag-agent.git
cd rag-agent
```

### Install Dependencies

```bash
# Install backend dependencies
pip install -r requirements.txt

# Install frontend dependencies
pip install -r requirements_streamlit.txt

# Or install all at once
pip install -r requirements.txt -r requirements_streamlit.txt
```

### Set Up Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

## 🚀 Usage

### Running Locally

**Step 1: Start the FastAPI Backend**

Open a terminal and run:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

**Step 2: Start the Streamlit Frontend**

Open another terminal and run:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Using the Application

1. **Choose Input Method**: Select File Upload, URL, or Text Input in the sidebar
2. **Upload Content**: Upload your document, paste a URL, or enter text
3. **Process**: Click the "Process" button to analyze your content
4. **Ask Questions**: Type your questions in the chat interface
5. **Get Answers**: Receive AI-powered answers from your documents or web search
6. **Add Context** (Optional): Add more sources to enrich your knowledge base

## 📚 API Documentation

Once the FastAPI backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/upload` | POST | Upload files (PDF, images) |
| `/api/upload-json` | POST | Upload URL or text content |
| `/api/query` | POST | Query documents |
| `/api/search-web` | POST | Search the web |
| `/api/add-context` | POST | Add file context |
| `/api/add-context-json` | POST | Add URL/text context |
| `/api/sessions/{id}/summary` | GET | Get session summary |
| `/health` | GET | Health check |

## 🔧 Configuration

### Changing the LLM Model

Edit `main.py`:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # Change to gemini-1.5-pro for better quality
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)
```

### Adjusting Chunk Size

Edit `main.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for longer context
    chunk_overlap=200  # Increase for better continuity
)
```

### Customizing UI Colors

Edit `streamlit_app.py` CSS section:

```python
.user-message {
    background-color: #f0f2f6;  # Change background
    border-left-color: #1f77b4;  # Change border
    color: #1f1f1f;  # Change text color
}
```

## 📁 Project Structure

```
rag-agent/
├── app.py                      # FastAPI backend
├── main.py                     # Core RAG logic (LangChain)
├── streamlit_app.py            # Streamlit frontend
├── requirements.txt            # Backend dependencies
├── requirements_streamlit.txt  # Frontend dependencies
├── .env                        # Environment variables (not in repo)
├── README.md                   # This file
└── screenshots/                # UI screenshots
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Known Issues

- Some websites with strong Cloudflare protection may not load (working on solution)
- Large PDFs (>50MB) may take longer to process
- Session data is stored in-memory (use Redis for production)

## 🗺️ Roadmap

- [ ] Add support for more document types (Word, Excel, PowerPoint)
- [ ] Implement conversation memory for follow-up questions
- [ ] Add user authentication and session persistence
- [ ] Deploy to cloud platforms (AWS, GCP, Azure)
- [ ] Add multi-language support
- [ ] Implement document comparison feature
- [ ] Add export chat history functionality

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the amazing framework
- [Google Gemini](https://deepmind.google/technologies/gemini/) for the powerful AI model
- [Streamlit](https://streamlit.io/) for the beautiful UI framework
- [FastAPI](https://fastapi.tiangolo.com/) for the modern API framework

## 💬 Support

If you have any questions or issues, please:
- Open an issue on [GitHub Issues](https://github.com/ayushaggarwalx/rag-agent/issues)
- Reach out on [Twitter/X](https://x.com/_AyushAggarwal)
- Email: ayushaggarwalx@gmail.com

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ayushaggarwalx/rag-agent&type=Date)](https://star-history.com/#AyushAggarwalx/rag-agent&Date)

---

**Made with ❤️ by [Ayush Aggarwal](https://github.com/ayushaggarwalx)**

If you find this project useful, please consider giving it a ⭐!
