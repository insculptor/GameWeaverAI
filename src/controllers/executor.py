"""
####################################################################################
#####                File: src/controllers/executor.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/07/2024                              #####
#####               Class to Manage Game Flow for GameWeaverAI                 #####
####################################################################################
"""
import logging
import os
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH","/app")
sys.path.append(ROOT_PATH)
print(f"[INFO]: {ROOT_PATH=}")
config_path = os.path.join(ROOT_PATH, 'config.yaml')
with open(config_path, 'r') as f:
        config =  yaml.safe_load(f)



from src.controllers.prompts import generate_game_rules_prompt
from src.models.llm_engine import LLMService
from src.rag.ingest_data import RAGIngestor
from src.rag.retrieve_data import RAGRetriever

# Setting up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

class GameFlow:
    def __init__(self):
        """
        Initializes GameFlow class with RAGRetriever, RAGIngestor, and LLMService instances.
        """
        self.retriever = RAGRetriever()
        self.ingestor = RAGIngestor()
        self.llm_service = LLMService()

    def play_game(self, game_name: str):
        """
        Manages the game flow when a user enters the game name and presses the 'Play' button.

        Args:
            game_name (str): The name of the game the user wants to play.
        
        Returns:
            str: The Python code for the game to be rendered on the UI.
        """
        logging.info(f"User selected the game: {game_name}")
        
        # 1. Search for matching game in the vector database
        game_metadata = self.retriever.fetch_document_metadata_by_name(game_name)
        
        if game_metadata:
            logging.info(f"Game '{game_name}' found in vector database. Retrieving metadata...")
            logging.info(f"Metadata retrieved for '{game_name}': {game_metadata}")
        else:
            logging.info(f"Game '{game_name}' not found in vector database. Generating game rules using LLM...")
            # 2. Generate game rules using LLMService if not found
            game_rules_prompt = generate_game_rules_prompt(game_name)
            logging.info(f"Generating game rules for '{game_name}' using LLMService.")
            logging.info(f"Game rules prompt: {game_rules_prompt}")
            game_rules = self.llm_service.generate_rules(game_rules_prompt)
            logging.info(f"Game rules generated for '{game_name}': {game_rules}")
            
            # Ingest generated rules into vector database
            logging.info(f"Ingesting generated game rules for '{game_name}' into the vector database.")
            self.ingestor.ingest_game_rules(game_name, game_rules)
            logging.info(f"Game rules ingested successfully for '{game_name}'.")
            
            # Retrieve metadata after ingestion
            game_metadata = self.retriever.fetch_document_metadata_by_name(game_name)
            logging.info(f"Metadata retrieved for '{game_name}' after ingestion: {game_metadata}")
        
        if not game_metadata:
            logging.error(f"Failed to retrieve metadata for '{game_name}' after ingestion.")
            return None

        # 3. Generate game code using the metadata
        logging.info(f"Generating code for the game '{game_name}' using LLMService.")
        game_code_prompt = self.retriever.get_metadata_prompt(game_metadata['ID'])
        logging.info(f"Game code prompt for '{game_name}': {game_code_prompt}")
        generated_code = self.llm_service.generate_code(game_code_prompt)
        
        if generated_code:
            logging.info(f"Game code generated for '{game_name}' and ready for rendering on the UI.")
            logging.debug(f"Generated code: {generated_code}")
        else:
            logging.error(f"Failed to generate game code for '{game_name}'.")
        
        return generated_code

# Running the GameFlow for "Tic Tac Toe"
if __name__ == "__main__":
    game_name = "Pytorch"
    game_flow = GameFlow()
    
    logging.info(f"Starting game flow for '{game_name}'...")
    game_code = game_flow.play_game(game_name)
    
    if game_code:
        print(f"Generated Python Code for '{game_name}':\n")
        print(game_code)
    else:
        print(f"Failed to generate code for '{game_name}'.")
