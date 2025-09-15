# from langchain_core.messages import AIMessage , HumanMessage
# import os
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv


# load_dotenv()
# model = ChatGroq(model="llama3-70b-8192")

# # main.py (continued)
# from langchain_core.prompts import ChatPromptTemplate

# prompt = ChatPromptTemplate.from_messages([
#     ("system", "You are a helpful assistant. I will be providing you a Simple Image of MRI you have to check if is is Severe and Answer the user's Question whatver user may ask"),
#     ("human", "What is Mri and Glioma Tumor")
# ])

# chain = prompt | model
# response = chain.invoke("What did the user ask ?")
# print(response.content)



import streamlit as st
import os
import base64
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import sys
print(sys.executable)


# --- 1. SETUP AND CONFIGURATION ---

# Load environment variables from a .env file
load_dotenv()

# Set Streamlit page configuration
st.set_page_config(page_title="Tumor Analysis Chatbot", layout="wide")
st.title(" Tumor Analysis Chatbot")

# --- 2. CORE CHATBOT ENGINE ---

# Use Streamlit's caching to initialize the model and chain only once.
@st.cache_resource
def get_chat_engine():
    """Initializes the Groq model and the LangChain runnable chain with memory."""
    
    # Initialize the Groq Vision model
    model = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
    
    # This dictionary will store chat histories for different sessions.
    # In a real-world app, you'd replace this with a persistent database (e.g., Redis, Postgres).
    store = {}

    def get_session_history(session_id: str):
        """Fetches or creates a chat history for a given session ID."""
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # Create the prompt template with a placeholder for chat history
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized medical assistant. Analyze the provided medical image and answer the user's questions about it. Be clear, concise, and helpful. Do not provide a diagnosis, but explain what you see in the image."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    # Create the runnable chain and wrap it with memory management
    chain_with_history = RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    return chain_with_history

# --- 3. STREAMLIT UI AND INTERACTION ---

# Get the chat engine
chat_engine = get_chat_engine()

# Initialize chat history in Streamlit's session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Handle both text and image content in user messages
        if "image" in message:
            st.image(message["image"], caption="Uploaded Image", width=200)
        st.markdown(message["content"])

# --- Sidebar for Image Upload ---
with st.sidebar:
    st.header("Upload Image")
    uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

# --- Main Chat Logic ---
if prompt := st.chat_input("Ask a question about the image..."):
    # Add user's message to session state and display it
    user_message = {"role": "user", "content": prompt}
    
    # Handle the image if it's uploaded with the first prompt
    if uploaded_file and not any("image" in m for m in st.session_state.messages):
         # Read the image and add it to the message
        image_bytes = uploaded_file.getvalue()
        user_message["image"] = image_bytes
        
        # Convert image to base64
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Prepare the multimodal input for the model
        model_input = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"},
            ]
        )
    else:
        # If no new image, send only the text prompt
        model_input = prompt

    st.session_state.messages.append(user_message)
    with st.chat_message("user"):
        if "image" in user_message:
            st.image(user_message["image"], caption="Uploaded Image", width=200)
        st.markdown(user_message["content"])

    # Get and display the AI's response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Invoke the chat engine
            response = chat_engine.invoke(
                {"input": model_input},
                config={"configurable": {"session_id": "main_session"}}
            )
            ai_response = response.content
            st.markdown(ai_response)
    
    # Add AI's response to session state
