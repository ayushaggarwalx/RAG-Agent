import sys
import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize LangChain components
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.7
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.environ["GEMINI_API_KEY"]
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


def get_gemini_model(multimodal=False):
    """Get Gemini model for direct API calls (for image processing)"""
    model = "gemini-1.5-flash"
    return genai.GenerativeModel(model)


def load_pdf(file_path: str):
    """Load PDF using PyMuPDF"""
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_image(image_path: str):
    """Extract text from image using Gemini Vision"""
    image = Image.open(image_path)
    prompt = "Extract all readable text from this image."
    gemini = get_gemini_model(multimodal=True)
    response = gemini.generate_content([prompt, image])
    return [Document(page_content=response.text, metadata={"source": image_path})]


def load_text(text: str):
    """Create document from plain text"""
    return [Document(page_content=text, metadata={"source": "text_input"})]


def load_url(url: str):
    """Load content from URL"""
    loader = WebBaseLoader(url)
    documents = loader.load()
    return documents


def load_from_type(input_type: str, value: str):
    """Load documents based on input type"""
    if input_type == "pdf":
        return load_pdf(value)
    elif input_type == "image":
        return load_image(value)
    elif input_type == "url":
        return load_url(value)
    elif input_type == "text":
        return load_text(value)
    else:
        raise ValueError(f"Unsupported input type: {input_type}")


def build_qa_chain(documents):
    """Build a RetrievalQA chain from documents"""
    # Split documents into chunks
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # Create custom prompt
    prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer or if the information is not in the context, say "The provided text does not contain information to answer this question."

Context: {context}

Question: {question}

Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    
    return qa_chain


def generate_summary(documents, input_type):
    """Generate a summary of the documents"""
    try:
        combined_text = ""
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
        
        if len(combined_text.strip()) < 100:
            return combined_text.strip()
        
        if input_type == "image":
            prompt = "Provide a brief 2-3 sentence summary of the text extracted from this image:"
        elif input_type == "pdf":
            prompt = "Provide a brief 2-3 sentence summary of this PDF document:"
        elif input_type == "url":
            prompt = "Provide a brief 2-3 sentence summary of this webpage content:"
        elif input_type == "mixed":
            prompt = "Provide a brief 2-3 sentence summary of this combined content from multiple sources:"
        else:
            prompt = "Provide a brief 2-3 sentence summary of this text:"
        
        full_prompt = f"{prompt}\n\n{combined_text[:2000]}..."
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"Could not generate summary: {str(e)}"


def generate_preview(documents, source_description):
    """Generate a brief preview of document content for display"""
    try:
        combined_text = ""
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
        
        preview_text = combined_text.strip()[:300]
        if len(combined_text.strip()) > 300:
            preview_text += "..."
        
        return {
            'source': source_description,
            'text_preview': preview_text,
            'character_count': len(combined_text.strip()),
            'document_count': len(documents)
        }
    except Exception as e:
        return {
            'source': source_description,
            'text_preview': f"Error generating preview: {str(e)}",
            'character_count': 0,
            'document_count': len(documents) if documents else 0
        }


def search_web_with_google(query):
    """Search the web using Gemini with Google Search grounding"""
    try:
        from google.genai import types
        
        google_search_tool = types.Tool(
            google_search=types.GoogleSearch()
        )
        
        gemini_model = get_gemini_model()
        search_prompt = f"Search the web for: {query}. Provide a concise answer based on the search results."
        
        response = gemini_model.generate_content(
            search_prompt,
            tools=[google_search_tool]
        )
        return response.text
    except Exception as e:
        error_str = str(e)
        if "429" in error_str and "RESOURCE_EXHAUSTED" in error_str:
            return "quota_exceeded"
        elif "quota" in error_str.lower() or "limit" in error_str.lower():
            return "quota_exceeded"
        else:
            return f"search_error: {error_str}"


def answer_with_gemini(query):
    """Answer question using Gemini's general knowledge"""
    try:
        response = llm.invoke(query)
        return response.content
    except Exception as e:
        return f"Gemini search failed: {str(e)}"


def is_answer_not_found(response_text):
    """Check if the answer indicates information was not found"""
    not_found_phrases = [
        "does not provide",
        "does not give", 
        "does not contain",
        "does not mention",
        "not found in",
        "no information",
        "cannot find",
        "doesn't provide",
        "doesn't give",
        "doesn't contain",
        "doesn't mention",
        "cannot be answered",
        "not available",
        "not provided",
        "from the given",
        "given context",
        "given text",
        "provided text",
        "available in the",
        "insufficient information"
    ]
    response_lower = response_text.lower()
    return any(phrase in response_lower for phrase in not_found_phrases)


def main():
    """Main CLI interface"""
    if len(sys.argv) < 3:
        print("Usage: python main.py [pdf|image|url|text] <file_path|image_path|url|text>")
        return

    input_type = sys.argv[1]
    input_value = sys.argv[2]

    docs = load_from_type(input_type, input_value)
    qa_chain = build_qa_chain(docs)

    # Generate and show summary
    print("📄 Processing your content...\n")
    summary = generate_summary(docs, input_type)
    print("📋 Summary:")
    print(f"{summary}\n")
    print("─" * 50)
    print("Ask a question (or type 'exit' or 'quit' to stop):")
    
    while True:
        try:
            query = input("> ")
            if query.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break
        
        result = qa_chain.invoke({"query": query})
        response_text = result["result"]
        print(response_text)
        
        if is_answer_not_found(response_text):
            print("\n⚠️  Information not found in the provided text.")
            try:
                web_search = input("Would you like to search the web instead? (y/n): ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Goodbye!")
                break
            
            if web_search in ['y', 'yes']:
                print("\n🔍 Searching Google...")
                google_result = search_web_with_google(query)
                
                if google_result == "quota_exceeded":
                    print("⚠️  Google Search quota exceeded. Using Gemini's general knowledge instead...")
                    gemini_result = answer_with_gemini(query)
                    print(f"\n💡 Gemini says: {gemini_result}")
                elif google_result.startswith("search_error:"):
                    print("🤖 Google search encountered an error. Let me try with Gemini's general knowledge...")
                    gemini_result = answer_with_gemini(query)
                    print(f"\n💡 Gemini says: {gemini_result}")
                else:
                    print(f"\n🌐 Google search result: {google_result}")
            print()


if __name__ == "__main__":
    main()