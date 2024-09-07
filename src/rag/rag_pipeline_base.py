"""
####################################################################################
#####                File: src/rag/rag_pipeline_base.py                        #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####         Base Class to Manage the RAG Pipeline using ChromaDB             #####
####################################################################################
"""
import json
import logging
import os

import PyPDF2
import torch
import yaml
from chromadb import PersistentClient
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.models.hf_models_manager import HFModelsManager

# Load environment variables
load_dotenv()

class RAGPipeline:
    """Base class for ChromaDB-based retrieval-augmented generation pipeline."""

    def __init__(self):
        ROOT_PATH = os.getenv("ROOT_PATH")
        config_path = os.path.join(ROOT_PATH, 'config.yaml')

        # Load config.yaml file
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load debug level from config
        debug_level = config["logging"].get("DEBUG_LEVEL", "INFO")
        logging.basicConfig(
            level=getattr(logging, debug_level.upper(), logging.INFO),  # Use config level, default to INFO
            format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logging.info("Initializing RAGPipeline...")

        # Set paths from environment variables or config
        self.DOCS_PATH = os.path.join(ROOT_PATH, os.getenv("DOCS_PATH"))
        self.VECTORSTORE_PATH = os.path.join(ROOT_PATH, os.getenv("VECTORSTORE_PATH"))
        self.MODELS_PATH = os.path.join(ROOT_PATH, os.getenv("MODELS_BASE_DIR"))
        self.EMBEDDING_MODEL = config["models"]["EMBEDDING_MODEL"]
        self.COLLECTION_NAME = config["vectorstore"]["CHROMADB_COLLECTION"]
        self.COLLECTION_MAPPING_PATH = os.path.join(self.VECTORSTORE_PATH, f"{self.COLLECTION_NAME}.json")
        self.SECTION_TITLES = config["game_rules"].get("section_titles", [])

        logging.info(f"Setting up paths: DOCS_PATH={self.DOCS_PATH}, VECTORSTORE_PATH={self.VECTORSTORE_PATH}, MODELS_PATH={self.MODELS_PATH}")
        logging.info(f"Using section titles: {self.SECTION_TITLES}")

        # Initialize model and tokenizer
        logging.info(f"Initializing HFModelsManager with model: {self.EMBEDDING_MODEL}")
        hf_manager = HFModelsManager(self.EMBEDDING_MODEL, model_path=self.MODELS_PATH)
        self.model, self.tokenizer = hf_manager.initialize_model()

        # Set up ChromaDB client and collection
        logging.info(f"Setting up ChromaDB client at {self.VECTORSTORE_PATH}")
        self.client = PersistentClient(path=self.VECTORSTORE_PATH)
        try:
            self.collection = self.client.get_collection(self.COLLECTION_NAME)
            logging.info(f"Loaded ChromaDB collection: {self.COLLECTION_NAME}")
        except Exception:
            logging.warning(f"Collection {self.COLLECTION_NAME} not found, creating new collection")
            self.collection = self.client.create_collection(self.COLLECTION_NAME)

    def get_collection_mapping(self):
        """Reads the Collection Master Mapping file."""
        logging.info("Reading collection mapping file...")
        if os.path.exists(self.COLLECTION_MAPPING_PATH):
            try:
                with open(self.COLLECTION_MAPPING_PATH, 'r') as f:
                    logging.debug(f"Loaded collection mapping from {self.COLLECTION_MAPPING_PATH}")
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to read collection mapping file: {e}")
        else:
            logging.warning(f"Collection mapping file not found at {self.COLLECTION_MAPPING_PATH}")
        return []

    def write_collection_mapping(self, game_rules):
        """Writes the updated game rules to the collection mapping JSON file."""
        logging.info(f"Writing updated collection mapping to {self.COLLECTION_MAPPING_PATH}")
        try:
            with open(self.COLLECTION_MAPPING_PATH, 'w') as f:
                json.dump(game_rules, f, indent=4)
            logging.debug("Successfully updated collection mapping")
        except Exception as e:
            logging.error(f"Failed to write collection mapping: {e}")

    def get_or_create_game_id(self, game_name):
        """Check if a game already exists in the collection mapping and return its ID, or create a new one."""
        logging.info(f"Checking or creating game ID for {game_name}")
        collection_mapping = self.get_collection_mapping()
        for game in collection_mapping:
            if game["Game Name"] == game_name:
                logging.debug(f"Found existing game ID: {game['ID']} for {game_name}")
                return game["ID"], collection_mapping
        new_id = len(collection_mapping) + 1
        logging.debug(f"Creating new game ID: {new_id} for {game_name}")
        return new_id, collection_mapping

    def read_pdf_sections(self, file_path):
        """Reads a PDF file and extracts text into sections."""
        logging.info(f"Reading PDF file at {file_path}")
        sections = {}
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                logging.debug(f"Extracted text from {file_path}")
        except Exception as e:
            logging.error(f"Failed to read PDF file {file_path}: {e}")
            return sections

        # Split sections based on known section titles from the config
        current_section = None
        for line in text.splitlines():
            if any(title in line for title in self.SECTION_TITLES):
                current_section = line.strip()
                sections[current_section] = []
            elif current_section:
                sections[current_section].append(line.strip())

        # Join lines for each section
        for section in sections:
            sections[section] = " ".join(sections[section])

        logging.info(f"Extracted sections: {list(sections.keys())}")
        return sections

    def preprocess_and_split_sections(self, sections, chunk_size=300, chunk_overlap=50):
        """Splits sections into chunks using RecursiveCharacterTextSplitter."""
        logging.info(f"Splitting sections into chunks with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = {}
        for section_name, section_text in sections.items():
            chunks[section_name] = text_splitter.split_text(section_text)
            logging.debug(f"Split section '{section_name}' into {len(chunks[section_name])} chunks")
        return chunks

    def create_embeddings(self, chunks):
        """Creates embeddings for the text chunks using the HF model."""
        logging.info("Creating embeddings for text chunks")
        embeddings = {}
        for section_name, chunk_list in chunks.items():
            embeddings[section_name] = []
            for i, chunk in enumerate(chunk_list):
                logging.debug(f"Creating embedding for chunk {i} in section '{section_name}'")
                inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
                    embeddings[section_name].append(embedding)
        logging.info("Embeddings created successfully")
        return embeddings
