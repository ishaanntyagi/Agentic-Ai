import os
from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

# Load API Keys
load_dotenv()

app = Flask(__name__)

# Global Storage (In-memory for demo)
chat_session = {
    "history": [],
    "patient_context": ""
}

# --- MODEL FACTORY ---
def get_model(model_choice):
    if model_choice == 'groq':
        # Using a fast, versatile model on Groq
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"), 
            model="llama-3.3-70b-versatile"
        )
    else:
        # Using local Ollama
        return ChatOllama(model="llama3.2")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_consultation', methods=['POST'])
def start_consultation():
    """Initializes the chat with patient details."""
    data = request.json
    
    # Construct the System Context
    context_str = (
        f"Patient Name: {data.get('name')}\n"
        f"Age: {data.get('age')}\n"
        f"Issue: {data.get('issue')}\n"
        f"Duration: {data.get('duration')}\n"
        f"Severity: {data.get('severity')}\n"
        f"Mental State: {data.get('mental_state')}\n"
        f"Hospital: {data.get('hospital')} ({data.get('area')})\n"
        f"Treating Dr: {data.get('dr_name')}"
    )
    
    # Reset Session
    chat_session["history"] = [
        SystemMessage(content=f"You are a compassionate medical AI assistant. Here is the patient's intake form:\n{context_str}\n\nUse this information to provide personalized advice. Be professional but empathetic.")
    ]
    chat_session["patient_context"] = context_str
    
    return jsonify({"status": "Consultation Started"})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    model_choice = request.json.get('model', 'groq') # Default to Groq

    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    try:
        # 1. Add User Message
        chat_session["history"].append(HumanMessage(content=user_input))
        
        # 2. Select Model
        llm = get_model(model_choice)
        
        # 3. Generate Response
        response = llm.invoke(chat_session["history"])
        
        # 4. Add AI Response
        chat_session["history"].append(AIMessage(content=response.content))

        return jsonify({'response': response.content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_history', methods=['GET'])
def get_history():
    """Returns the chat history in a readable format."""
    formatted_history = []
    for msg in chat_session["history"]:
        if isinstance(msg, HumanMessage):
            formatted_history.append({"role": "User", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted_history.append({"role": "AI", "content": msg.content})
        # We skip SystemMessage for the UI history view to keep it clean
            
    return jsonify({"history": formatted_history})

if __name__ == '__main__':
    app.run(debug=True, port=5000)