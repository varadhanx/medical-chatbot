from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings
)

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY or OPENAI_API_KEY in .env file")

# Set environment variables for downstream libraries
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load and preprocess data
extracted_data = load_pdf_file(data='data/')
filtered_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filtered_data)

# Load embeddings model
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Check if index exists, else create it
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=384,  # matches sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# Get reference to index
index = pc.Index(index_name)

# Store documents in Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print(f"Successfully loaded {len(text_chunks)} chunks into Pinecone index '{index_name}'.")
