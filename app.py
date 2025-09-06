from flask import Flask, render_template, request
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# ✅ FIX: Import helper and prompt correctly
try:
    from src.helper import download_hugging_face_embeddings
    from src.prompt import system_prompt
except ModuleNotFoundError:
    from helper import download_hugging_face_embeddings
    from prompt import system_prompt


app = Flask(__name__)

# ✅ Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ✅ Get embeddings
embeddings = download_hugging_face_embeddings()

# ✅ Load Pinecone Index
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# ✅ Setup Retrieval & Chat
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
chat_model = ChatOpenAI(model="gpt-4o")
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return "No message received", 400

    print(f"User: {msg}")
    response = rag_chain.invoke({"input": msg})
    print(f"Response: {response.get('answer', '')}")
    return str(response.get("answer", "No response"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
