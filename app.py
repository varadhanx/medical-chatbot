# OLD (causes ModuleNotFoundError)
from helper import download_hugging_face_embeddings
from prompt import system_prompt

# NEW (correct path)
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
import os

# LangChain and Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

# -----------------------
# App Initialization
# -----------------------
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise EnvironmentError("Missing PINECONE_API_KEY or OPENAI_API_KEY in your .env file")

# Set for downstream libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# -----------------------
# Pinecone & LangChain Setup
# -----------------------
print("Loading embeddings model...")
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"
print(f"Connecting to Pinecone index: {index_name}")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_model = ChatOpenAI(model="gpt-4o")

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# -----------------------
# Routes
# -----------------------
@app.route("/")
def index():
    """Renders chat UI"""
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    """Handles chat messages via POST"""
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    try:
        print(f"User: {msg}")
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn’t find an answer.")
        print(f"Bot: {answer}")
        return jsonify({"answer": answer})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Something went wrong on the server"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

