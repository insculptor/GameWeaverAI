"""
####################################################################################
#####                    File: src/rag/retrieve_data.py                        #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####               Class to Manage Retrival from RAG Pipeline                 #####
####################################################################################
"""
from src.rag.rag_pipeline_base import RAGPipeline


class RAGRetriever(RAGPipeline):
    """Class to handle metadata retrieval."""

    def fetch_document_metadata(self, game_id):
        """Fetches all metadata and chunks related to a specific game ID from ChromaDB."""
        game_rules = self.get_collection_mapping()

        # Find the game entry with the specified ID
        game_entry = next((rule for rule in game_rules if str(rule["ID"]) == str(game_id)), None)
        if not game_entry:
            print(f"No game found with ID: {game_id}")
            return None

        # Get the GAME_NAME from the entry
        game_name = game_entry["Game Name"]

        # Fetch all documents from the collection
        all_documents = self.collection.get()

        if not all_documents:
            print(f"No documents retrieved from ChromaDB for Game_Name: {game_name}")
            return None

        # Initialize the response dictionary
        response = {
            "ID": game_id,
            "Game_Name": game_name,
            "Chunks": {}
        }

        search_pattern = f"{game_name}_"
        if all_documents.get("ids") and all_documents.get("metadatas"):
            for i, chunk_id in enumerate(all_documents["ids"]):
                if chunk_id.startswith(search_pattern):
                    section_name = all_documents["metadatas"][i].get("Section_name", "Unknown Section")
                    chunk_text = all_documents["metadatas"][i].get("Chunk_Text", "No Chunk Available")
                    full_text = all_documents["metadatas"][i].get("Text", "No Full Text Available")

                    if chunk_id not in response["Chunks"]:
                        response["Chunks"][chunk_id] = {
                            "Section_name": section_name,
                            "Text": full_text,
                            "Chunk_Text": [chunk_text]
                        }
                    else:
                        response["Chunks"][chunk_id]["Chunk_Text"].append(chunk_text)

        return response
