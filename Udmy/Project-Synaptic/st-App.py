import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
import tempfile

# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

# Set the title for the Streamlit app
st.set_page_config(page_title="Medical RAG Chatbot", layout="wide")
st.title("Medical Report RAG Chatbot")
st.info("Upload a medical report PDF, choose an AI model, and press 'Process Document' to begin.")


# --- 2. CORE RAG LOGIC (CACHED FUNCTIONS) ---

@st.cache_resource
def create_vector_store(pdf_file):
    """Creates a FAISS vector store from an uploaded PDF file."""
    if pdf_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
            vector_store = FAISS.from_documents(chunks, embeddings)
            
            os.remove(tmp_file_path)
            return vector_store
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None
    return None

@st.cache_resource
def get_llm(model_choice):
    """Initializes and caches the selected LLM."""
    if model_choice == "Ollama":
        return Ollama(model="llama3")
    elif model_choice == "Groq":
        return ChatGroq(model="qwen/qwen1.5-72b-chat", groq_api_key=groq_api_key)
    else: # Perplexity
        return ChatPerplexity(model="llama-3-sonar-large-32k-online", api_key=perplexity_api_key)

@st.cache_resource
def get_rag_chain(_vector_store, _llm):
    """Creates and caches the RAG chain."""
    if _vector_store is None or _llm is None:
        return None
        
    retriever = _vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        You are an advanced AI medical assistant. Your primary goal is to answer questions using the provided medical report in the <context>. However, you may enrich your answers with general medical knowledge to provide clarity.
        **Core Operating Procedure:**
        1.  **Prioritize the Document:** Always start by looking for the answer in the provided <context>.
        2.  **Enrich with General Knowledge:** If the document mentions a specific medical term, explain it using the context first, then add a general definition.
        3.  **Attribute Your Sources:** Clearly state if information is from the report ("According to the report...") or general knowledge ("For context...").
        4.  **Medical Disclaimer:** Never provide a diagnosis or medical advice. Always recommend consulting a healthcare professional.
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """
        **Context from the medical report:**
        <context>
        {context}
        </context>
        **User's Question:**
        {input}
        """),
    ])

    question_answer_chain = create_stuff_documents_chain(_llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain


# --- 3. SESSION STATE MANAGEMENT ---

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None


# --- 4. UI COMPONENTS (SIDEBAR) ---

with st.sidebar:
    st.header("Setup")
    uploaded_file = st.file_uploader("Upload your Medical Report (PDF)", type="pdf")
    
    model_choice = st.selectbox(
        "Choose your AI Model:",
        ("Ollama", "Groq", "Perplexity"),
        disabled=(uploaded_file is None)
    )

    if st.button("Process Document"):
        if uploaded_file:
            with st.spinner("Processing document... This may take a moment."):
                vector_store = create_vector_store(uploaded_file)
                if vector_store:
                    llm = get_llm(model_choice)
                    st.session_state.rag_chain = get_rag_chain(vector_store, llm)
                    st.session_state.chat_history = [AIMessage(content="Document processed. How can I help you with this report?")]
                    st.success("Document processed and ready!")
                else:
                    st.error("Failed to process the document.")
        else:
            st.warning("Please upload a PDF file first.")

# --- 5. CHAT INTERFACE ---

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)

user_query = st.chat_input(
    "Ask a question about the document...", 
    disabled=not st.session_state.rag_chain
)

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        with st.spinner("Thinking..."):
            response = st.session_state.rag_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            ai_response = response["answer"]
            st.markdown(ai_response)
    
    st.session_state.chat_history.append(AIMessage(content=ai_response))

