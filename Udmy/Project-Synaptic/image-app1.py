import os
import base64
from dotenv import load_dotenv

# langchain_groq ChatGroq client
from langchain_groq import ChatGroq

# langchain-core message / prompt helpers
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# history runnable
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

def get_chat_engine():
    """
    Create a runnable chat engine that accepts prompt -> model and stores per-session history.
    NOTE: Replace the model_name below with a vision-capable model available in your Groq account.
    """
    # Use a vision-capable model name. Replace with the exact name your Groq account supports.
    # If the one below isn't available for you, change it to the model Groq provides for vision.
    model = ChatGroq(model_name="meta-llama/llama-4-scout-17b-16e-instruct")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful medical assistant. Your task is to analyze medical images. "
         "Describe the visual contents of the image in detail. Note any anomalies or unusual features you observe. "
         "Do not provide a diagnosis, but describe what you see visually."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    runnable_with_history = RunnableWithMessageHistory(
        prompt | model,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )
    return runnable_with_history

def main():
    print("--- Medical Image Analysis CLI Chatbot ---")

    chat_engine = get_chat_engine()

    # Change this to the local path of the image you want to test
    image_path = r"C:\Users\ishaa\OneDrive\Desktop\Projects\Medical_Image_Captioning\testing images\glioma\Tr-me_0031.jpg"
    print(f"Using preset image path: {image_path}")

    if not os.path.exists(image_path):
        print("Error: File not found. Please check the path in the script.")
        return

    # Make sure your env contains the keys Groq/OpenAI if required:
    # GROQ_API_KEY, OPENAI_API_KEY, etc. (load_dotenv above will load from .env)
    print("Image loaded. You can now start asking questions.")
    print("Type 'exit' or 'quit' to end the conversation.")

    first_turn = True

    while True:
        user_prompt = input("\nYou: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("AI: Goodbye!")
            break

        content_parts = []
        # Always include the text prompt
        content_parts.append({"type": "text", "text": user_prompt})

        # Send the image only on the first turn (you can change that behavior)
        if first_turn:
            print("AI: Processing image with your first question...")
            try:
                with open(image_path, "rb") as image_file:
                    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

                # IMPORTANT: image_url must be a dict with "url" key (data URI allowed)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                })

                first_turn = False
            except Exception as e:
                print(f"AI: Sorry, I had trouble reading the image file. Error: {e}")
                continue

        # Construct the HumanMessage with the list of content parts
        model_input = HumanMessage(content=content_parts)

        try:
            response = chat_engine.invoke(
                {"input": model_input},
                config={"configurable": {"session_id": "cli_session"}}
            )
            # Response usually in response.content
            print(f"\nAI: {response.content}")
        except Exception as e:
            print(f"AI: Error invoking model: {e}")
            # Optional: print full exception or re-raise during debugging
            # raise

if __name__ == "__main__":
    main()
