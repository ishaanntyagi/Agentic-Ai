from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
import os
import pathlib
import dotenv
from langchain_core.messages import AIMessage, HumanMessage
from pprint import pprint


from dotenv import load_dotenv
load_dotenv()

openai_api_key     = os.getenv("OPENAI_API_KEY")
gemini_api_key     = os.getenv("GEMINI_API_KEY")
groq_api_key       = os.getenv("GROQ_API_KEY")
langchain_api_key  = os.getenv("LANGCHAIN_API_KEY")
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")

#loading The PDF

loader = PyPDFLoader(r"C:\Users\ishaa\Downloads\sample_medical_report.pdf")
medical_report = loader.load()
print("medical report loaded")
print(f"loaded doc has : {len(medical_report)} Page")


#text Spiltter 
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 150,
    chunk_overlap = 20,
    length_function = len, #To count the CharS
)
chunks = splitter.split_documents(medical_report)
print("Chunks created")
print(f"{len(chunks)} : length of chunks")      


from langchain.embeddings import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'} 
encode_kwargs = {'normalize_embeddings': False}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embedding model loaded From Huggingface")


#The embeddings and the vectpor Db part 
db = FAISS.from_documents(chunks,embeddings)
db.save_local("faiss_index")
current_directory = os.getcwd()
index_path = os.path.join(current_directory, "faiss_index")
print(f"The Embeddings have been made @ {index_path}")



from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

print("Databade Loaded")



#Creating The context retriever 

retriever = db.as_retriever()
print(f"retriever created : \n {retriever}")


# Lets make The user choose whatever model he/she wantes to use: 
#function will be Createsd to let  User Choose between 
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAI
from langchain_perplexity import ChatPerplexity 
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage , HumanMessage
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
from langchain_perplexity import ChatPerplexity
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder 
from pprint import pprint


def get_llm():

    while True:
        user_choice = input("Enter the model you want to use For the ChatBot Purposes (ollama, groq, perplexity): ").lower()

        if user_choice == "ollama":
            print("System : Starting Ollama model")
            return Ollama(model="llama3")
        
        elif user_choice == "groq":
            print("System : Starting model(qwen-32b)...")
            return ChatGroq(model="qwen/qwen3-32b", groq_api_key=os.getenv("GROQ_API_KEY"))
            
        elif user_choice == "perplexity":
            print("System : Starting model FROM PERPLEXITY")
            return ChatPerplexity(model="llama-3-sonar-large-32k-online", perplexity_api_key=os.getenv("PERPLEXITY_API_KEY"))
            
        else:
            print("INVALID CHOICE")

    print("System Started Successfully")

llm = get_llm()


prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an advanced AI medical assistant. Your primary goal is to answer questions using the provided medical report in the <context>. However, you may enrich your answers with general medical knowledge to provide clarity.

    **Your Core Operating Procedure:**
    1.  **Prioritize the Document:** Always start by looking for the answer in the provided <context>. This is your primary source of truth.
    2.  **Enrich with General Knowledge:** If the document mentions a specific medical term (e.g., "bradycardia"), first use the context to describe the patient's specific case. Then, you may add a general definition from your own knowledge.
    3.  **Attribute Your Sources (Crucial Rule):** You MUST clearly distinguish between information from the document and general knowledge. Use phrases like "According to the report..." or "For context, in general medicine...".
    4.  **Medical Disclaimer:** Never provide a direct diagnosis or medical advice. Always recommend consulting a qualified healthcare professional.
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


question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

print("System : ChainCreatedSuccessfully")


chat_history = []
print("Syetem : Starting Chatbot ")
print("Type 'exit' to end the conversation.")

# while True:
#     question = input(" Ask a question about the medical report: ")
#     if question.lower() == 'exit':
#         print(" Conversation ended Goodbye ")
#         break


#     response = rag_chain.invoke({
#         "input": question,
#         "chat_history": chat_history
#     })

#     print("Answer")
#     pprint(response["answer"])


#     chat_history.extend([
#         HumanMessage(content=question),
#         AIMessage(content=response["answer"]),
#     ])


while True:
    question = input("\nAsk a question about the medical report: ")
    if question.lower() == 'exit':
        print("Conversation ended. Goodbye!")
        break

    response = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    print("\n--- Answer ---")
    print(response["answer"])

    chat_history.extend([
        HumanMessage(content=question),
        AIMessage(content=response["answer"]),
    ])