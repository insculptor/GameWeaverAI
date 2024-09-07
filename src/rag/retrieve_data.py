"""
####################################################################################
#####                    File: src/rag/retrieve_data.py                        #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####               Class to Manage Retrieval from RAG Pipeline                #####
####################################################################################
"""

import logging

from src.controllers.prompts import generate_code_prompt
from src.rag.rag_pipeline_base import RAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to the console
)

class RAGRetriever(RAGPipeline):
    """Class to handle metadata retrieval from the RAG pipeline."""

    def fetch_document_metadata(self, game_id):
        """
        Fetches all metadata and chunks related to a specific game ID from ChromaDB 
        and returns them in a structured format.
        
        Args:
            game_id (str): The ID of the game to fetch metadata for.

        Returns:
            dict: A dictionary containing structured metadata where the keys are section names
                  and the values are dictionaries with 'Text' and 'Chunk_Text'.
        """
        logging.info(f"Fetching document metadata for game ID: {game_id}")

        # Get the collection mapping
        game_rules = self.get_collection_mapping()

        # Find the game entry with the specified ID
        game_entry = next((rule for rule in game_rules if str(rule["ID"]) == str(game_id)), None)
        if not game_entry:
            logging.error(f"No game found with ID: {game_id}")
            return None

        # Get the GAME_NAME from the entry
        game_name = game_entry["Game Name"]
        logging.debug(f"Found game: {game_name} with ID: {game_id}")

        # Fetch all documents from the collection
        all_documents = self.collection.get()
        if not all_documents:
            logging.error(f"No documents retrieved from ChromaDB for Game_Name: {game_name}")
            return None

        # Initialize the response dictionary with the desired structure
        response = {
            "ID": game_id,
            "Game_Name": game_name,
            "Overview": {"Text": "", "Chunk_Text": []},
            "Game Setup": {"Text": "", "Chunk_Text": []},
            "How to Play": {"Text": "", "Chunk_Text": []},
            "Winning the Game": {"Text": "", "Chunk_Text": []},
            "Game Strategy": {"Text": "", "Chunk_Text": []},
            "End of Game": {"Text": "", "Chunk_Text": []},
        }

        logging.debug(f"All Documents: {all_documents}")

        # Define search pattern to match the document IDs starting with the game name
        search_pattern = f"{game_name}_"
        if all_documents.get("ids") and all_documents.get("metadatas"):
            for i, chunk_id in enumerate(all_documents["ids"]):
                logging.debug(f"Checking chunk_id: {chunk_id}")
                if chunk_id.startswith(search_pattern):
                    # Get metadata for the current chunk
                    section_name = all_documents["metadatas"][i].get("Section_name", "Unknown Section")
                    chunk_text = all_documents["metadatas"][i].get("Chunk_Text", "No Chunk Available")
                    full_text = all_documents["metadatas"][i].get("Text", "No Full Text Available")

                    logging.debug(f"Found Section: {section_name}, Chunk: {chunk_text}")

                    if section_name in response:
                        # Populate the full section text only once
                        if not response[section_name]["Text"]:
                            response[section_name]["Text"] = full_text
                        # Append the chunk to the Chunk_Text list
                        response[section_name]["Chunk_Text"].append(chunk_text)
                    else:
                        logging.warning(f"Section {section_name} not found in response")
        else:
            logging.warning("No ids or metadatas found in the documents.")

        logging.info(f"Final Response Structure: {response}")

        return response

    def get_metadata_prompt(self, game_id):
        """
        Fetches metadata for a given game ID and prepares it for generating the code prompt.

        Args:
            game_id (str): The ID of the game for which metadata is to be retrieved.

        Returns:
            str: A formatted prompt ready to be used for generating code.
        """
        logging.info(f"Fetching metadata prompt for game ID: {game_id}")

        # Fetch the document metadata using the existing fetch_document_metadata function
        metadata = self.fetch_document_metadata(game_id)

        if metadata is None:
            logging.error(f"No metadata found for game ID: {game_id}")
            return None

        # Log the structured metadata
        logging.debug(f"Metadata: {metadata}")
        
        # Generate the code prompt
        code_prompt = generate_code_prompt(metadata)
        logging.info("Generated code prompt successfully.")
        logging.debug(f"Prompt: {code_prompt}")
        
        return code_prompt

