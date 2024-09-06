"""
####################################################################################
#####                File: src/rag/rag_pipeline_base.py                        #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####         Base Class to Manage the RAG Pipeline using ChromaDB             #####
####################################################################################
"""
import json
import os

import PyPDF2
import torch
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.models.models import HFModelsManager

# Load environment variables
load_dotenv()

class RAGPipeline:
    """Base class for ChromaDB-based retrieval-augmented generation pipeline."""

    def __init__(self):
        self.DOCS_PATH = os.path.join(os.getenv("ROOT_PATH"), os.getenv("DOCS_PATH"))
        self.VECTORSTORE_PATH = os.path.join(os.getenv("ROOT_PATH"), os.getenv("VECTORSTORE_PATH"))
        self.MODELS_PATH = os.path.join(os.getenv("ROOT_PATH"), os.getenv("MODELS_BASE_DIR"))
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
        self.COLLECTION_NAME = os.getenv("CHROMADB_COLLECTION")
        self.COLLECTION_MAPPING_PATH = os.path.join(self.VECTORSTORE_PATH, f"{self.COLLECTION_NAME}.json")

        # Initialize model and tokenizer
        hf_manager = HFModelsManager(self.EMBEDDING_MODEL, model_path=self.MODELS_PATH)
        self.model, self.tokenizer = hf_manager.initialize_model()

        # Set up ChromaDB client and collection
        self.client = PersistentClient(path=self.VECTORSTORE_PATH)
        try:
            self.collection = self.client.get_collection(self.COLLECTION_NAME)
        except Exception:
            self.collection = self.client.create_collection(self.COLLECTION_NAME)

    def get_collection_mapping(self):
        """Reads the Collection Master Mapping file."""
        if os.path.exists(self.COLLECTION_MAPPING_PATH):
            with open(self.COLLECTION_MAPPING_PATH, 'r') as f:
                return json.load(f)
        return []

    def write_collection_mapping(self, game_rules):
        """Writes the updated game rules to the collection mapping JSON file."""
        with open(self.COLLECTION_MAPPING_PATH, 'w') as f:
            json.dump(game_rules, f, indent=4)

    def get_or_create_game_id(self, game_name):
        """Check if a game already exists in the collection mapping and return its ID, or create a new one."""
        collection_mapping = self.get_collection_mapping()
        for game in collection_mapping:
            if game["Game Name"] == game_name:
                return game["ID"], collection_mapping
        new_id = len(collection_mapping) + 1
        return new_id, collection_mapping

    def read_pdf_sections(self, file_path):
        """Reads a PDF file and extracts text into sections."""
        sections = {}
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()

        section_titles = ["Overview", "Game Setup", "How to Play", "Winning the Game", "Game Strategy", "End of Game"]
        current_section = None
        for line in text.splitlines():
            if any(title in line for title in section_titles):
                current_section = line.strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line.strip())

        for section in sections:
            sections[section] = " ".join(sections[section])

        return sections

    def preprocess_and_split_sections(self, sections, chunk_size=300, chunk_overlap=50):
        """Splits sections into chunks using RecursiveCharacterTextSplitter."""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = {}
        for section_name, section_text in sections.items():
            chunks[section_name] = text_splitter.split_text(section_text)
        return chunks

    def create_embeddings(self, chunks):
        """Creates embeddings for the text chunks using the HF model."""
        embeddings = {}
        for section_name, chunk_list in chunks.items():
            embeddings[section_name] = []
            for chunk in chunk_list:
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
                    embeddings[section_name].append(embedding)
        return embeddings
