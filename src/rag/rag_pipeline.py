# src/rag/rag_pipeline.py

import os

import chromadb
import PyPDF2
import torch
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

# Load environment variables
load_dotenv()

# Initialize ChromaDB client
client = chromadb.Client()

# Path to the documents directory from .env
DOCS_PATH = os.getenv("DOCS_PATH")

# Load the embedding model and tokenizer from Hugging Face
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "WhereIsAI/UAE-Large-V1")
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
model = AutoModel.from_pretrained(EMBEDDING_MODEL)

def setup_chromadb():
    """Initialize or get the collection from ChromaDB."""
    if not client.collection_exists("game_rules"):
        collection = client.create_collection("game_rules")
    else:
        collection = client.get_collection("game_rules")
    return collection

def check_game_exists(game_name):
    """Check if the game exists in the ChromaDB collection."""
    collection = setup_chromadb()
    results = collection.search(query_text=game_name)
    if results and results["matches"]:
        return True, results["documents"][0]
    return False, None

def fetch_game_rules(game_name):
    """Fetch the rules of a game if it exists."""
    exists, document = check_game_exists(game_name)
    if exists:
        return document["text"]  # Return the game rules from the document
    else:
        return None

def read_pdf(file_path):
    """Reads the text from a PDF file."""
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500):
    """Splits the text into chunks of a specified size."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def create_embeddings(chunks):
    """Creates embeddings for the chunks of text using the Hugging Face model."""
    embeddings = []
    for chunk in chunks:
        # Tokenize the input text
        inputs = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Typically, the embeddings are taken from the last hidden state
            # Mean pool over the sequence dimension (token-level embeddings to sentence-level)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            embeddings.append(embedding)
    
    return embeddings

def load_data_into_chromadb(chunks, embeddings, file_path):
    """Loads the chunks and embeddings into ChromaDB."""
    collection = setup_chromadb()

    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Add document to ChromaDB
        collection.add(documents=[{
            "text": chunk,
            "embedding": embedding.tolist(),  # Convert the numpy array to list
            "metadata": {"source": os.path.basename(file_path), "chunk_id": i}
        }])

def ingest_game_rules(file_path=None):
    """Main function to ingest game rules from a PDF."""
    if file_path is None:
        file_path = os.path.join(DOCS_PATH, "tic_tac_toe.pdf")

    # Step 1: Read the PDF
    text = read_pdf(file_path)

    # Step 2: Chunk the text
    chunks = chunk_text(text)

    # Step 3: Create embeddings for each chunk
    embeddings = create_embeddings(chunks)

    # Step 4: Load the chunks and embeddings into ChromaDB
    load_data_into_chromadb(chunks, embeddings, file_path)
