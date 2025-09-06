import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv

# LangChain and Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Local imports
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    st.error("Missing PINECONE_API_KEY or OPENAI_API_KEY in your .env file")
    st.stop()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Initialize session state for chain and messages
if "rag_chain" not in st.session_state:
    with st.spinner("Loading embeddings model..."):
        embeddings = download_hugging_face_embeddings()

    index_name = "medical-chatbot"
    with st.spinner(f"Connecting to Pinecone index: {index_name}"):
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
    st.session_state.rag_chain = create_retrieval_chain(retriever, question_answer_chain)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
st.title("Medical Chatbot")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a medical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({"input": prompt})
                answer = response.get("answer", "Sorry, I couldn’t find an answer.")
            except Exception as e:
                answer = "Something went wrong. Please try again."
                st.error(f"Error: {e}")
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
