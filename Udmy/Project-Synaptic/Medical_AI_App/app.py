import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# LangChain & Vector Store Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# LLM Imports
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DB_FOLDER'] = 'db'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DB_FOLDER'], exist_ok=True)

# --- GLOBAL STATE (For Demo Only) ---
session_state = {
    "vectorstore": None,
    "patient_context": "",
    "chat_history": [],
    "llm": None
}

# Initialize Embeddings once (runs locally)
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_llm(model_type):
    """Selects the Model"""
    if model_type == 'groq':
        if not os.getenv('GROQ_API_KEY'):
            raise ValueError("GROQ_API_KEY missing in .env")
        # Using a reliable model ID
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
    elif model_type == 'ollama':
        return ChatOllama(model="llama3.2")
    else:
        # Default fallback
        return ChatGroq(model="llama-3.3-70b-versatile")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/initialize_session', methods=['POST'])
def initialize_session():
    """Step 1: Save Patient Details & Select Model"""
    data = request.json
    
    # 1. Set Model
    try:
        session_state["llm"] = get_llm(data.get('model'))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # 2. Build Patient Context String
    patient_info = data.get('patient_data', {})
    context_str = (
        f"PATIENT DETAILS:\n"
        f"Name: {patient_info.get('name')}\n"
        f"Age: {patient_info.get('age')}\n"
        f"Complaint: {patient_info.get('issue')}\n"
        f"History: {patient_info.get('history', 'None')}\n"
    )
    session_state["patient_context"] = context_str
    
    # 3. Reset History
    session_state["chat_history"] = []
    
    return jsonify({"status": "Session Initialized"})

@app.route('/upload_report', methods=['POST'])
def upload_report():
    """Step 2: Upload & Process PDF (RAG)"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No file selected (Skipping RAG)"})

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        # Process PDF
        loader = PyPDFLoader(save_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        
        # Create Vector Store
        session_state["vectorstore"] = Chroma.from_documents(
            documents=splits,
            embedding=embedding_function,
            persist_directory=app.config['DB_FOLDER']
        )
        return jsonify({"status": "PDF Processed Successfully"})
    
    except Exception as e:
        print(f"PDF Error: {e}")
        return jsonify({"error": "Failed to process PDF"}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Step 3: Chat with Context + RAG"""
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    if not session_state["llm"]:
        # Auto-initialize if forgotten (optional safety)
        try:
            session_state["llm"] = get_llm('groq')
        except:
            return jsonify({"error": "Session not initialized. Please fill intake form."}), 400

    # 1. Retrieve RAG Context (if PDF exists)
    rag_context = ""
    if session_state["vectorstore"]:
        # Retrieve top 3 chunks
        docs = session_state["vectorstore"].as_retriever(search_kwargs={"k": 3}).invoke(user_input)
        rag_context = "\n\n".join([d.page_content for d in docs])
        rag_context = f"\nMEDICAL REPORT CONTEXT:\n{rag_context}\n"

    # 2. Build Prompt
    system_msg = (
        "You are an expert Medical AI Assistant.\n"
        "Use the Patient Details and Medical Report Context below to answer.\n"
        "If the answer isn't in the report, use your general medical knowledge but mention that it's general advice At the END I need A severity SCORE AND HOW TUMOR MAY AFFECT THE PATEINT < THE SEVIERITY SCORE IS COMPULSORYYYYY.\n\n"
        f"{session_state['patient_context']}\n"
        f"{rag_context}"
    )

    # 3. Add to History & Invoke
    # We construct the message list dynamically to include the latest system prompt with context
    messages = [SystemMessage(content=system_msg)] + session_state["chat_history"] + [HumanMessage(content=user_input)]
    
    try:
        response = session_state["llm"].invoke(messages)
        
        # Update History
        session_state["chat_history"].append(HumanMessage(content=user_input))
        session_state["chat_history"].append(AIMessage(content=response.content))
        
        return jsonify({"response": response.content})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)