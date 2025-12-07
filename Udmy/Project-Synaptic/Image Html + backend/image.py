import os
import base64
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# 1. Setup
load_dotenv()
app = Flask(__name__)

# Initialize Groq (Using the vision model)
# Note: Using 'llama-3.2-11b-vision-preview' as it is standard for vision on Groq now.
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="meta-llama/llama-4-scout-17b-16e-instruct")
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    user_prompt = request.form.get('prompt', 'Describe this medical image in detail. Identify potential anomalies.At the END I need A severity SCORE AND HOW TUMOR MAY AFFECT THE PATEINT < THE SEVIERITY SCORE IS COMPULSORYYYYY')

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # 2. Process Image (Encode to Base64)
        image_data = file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        image_type = file.content_type  # e.g., image/jpeg

        image_data_uri = f"data:{image_type};base64,{image_base64}"

        # 3. Call Groq Vision Model
        message = HumanMessage(
            content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_uri}},
            ]
        )

        response = llm.invoke([message])
        
        # 4. Return the result
        return jsonify({'result': response.content})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)