"""
####################################################################################
#####                     File: src/rag/ingest_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####              Class to Ingest Data into from RAG Pipeline                 #####
####################################################################################
"""
import logging
import os

from src.rag.rag_pipeline_base import RAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see verbose output
    format="%(filename)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to the console
)

class RAGIngestor(RAGPipeline):
    """Class to handle document ingestion into ChromaDB."""

    def ingest_document(self, file_path):
        """
        Ingests each section of the PDF as separate documents with embeddings into ChromaDB.
        
        Args:
            file_path (str): Path to the PDF file to be ingested.
        """
        logging.info(f"Starting ingestion for file: {file_path}")

        # Extract the game name from the file path (assumes PDF file format)
        game_name = os.path.basename(file_path).replace(".pdf", "")
        game_id, game_rules = self.get_or_create_game_id(game_name)

        logging.debug(f"Game ID: {game_id}, Game Name: {game_name}")

        # Read sections from the PDF
        sections = self.read_pdf_sections(file_path)
        if not sections:
            logging.error(f"No sections found in {file_path}")
            return

        logging.info(f"Sections found: {list(sections.keys())}")

        # Preprocess and split each section into chunks
        chunks = self.preprocess_and_split_sections(sections, chunk_size=300, chunk_overlap=50)
        if not chunks:
            logging.error(f"No chunks created for {file_path}")
            return

        logging.debug(f"Chunks created: {chunks}")

        # Create embeddings for the chunks
        embeddings = self.create_embeddings(chunks)
        logging.debug("Embeddings generated for chunks")

        # Ingest each chunk with metadata
        for section_name, chunk_list in chunks.items():
            full_section_text = sections[section_name]

            for i, chunk in enumerate(chunk_list):
                # Construct a unique document ID using game name and chunk index
                document_id = f"{game_name}_{section_name.replace(' ', '_')}_{i}"

                # Metadata for each chunk, including the full section text and chunk text
                document_metadata = {
                    "Game_ID": str(game_id),
                    "Game_Name": game_name,
                    "Section_name": section_name.strip(':'),  # Remove any trailing colons
                    "Text": full_section_text,  # Store the full section text
                    "Chunk_Text": chunk  # Store the chunk text
                }

                # Logging the document details
                logging.info(f"Ingesting Document ID: {document_id}")
                logging.debug(f"Metadata: {document_metadata}")

                # Upsert the document into ChromaDB (each chunk is treated as a separate document)
                self.collection.upsert(
                    ids=[document_id],
                    documents=[chunk],
                    embeddings=[embeddings[section_name][i].tolist()],
                    metadatas=[document_metadata]
                )

        # Update the JSON file to track the ingested game
        if not any(rule["Game Name"] == game_name for rule in game_rules):
            game_rules.append({"ID": game_id, "Game Name": game_name})
            self.write_collection_mapping(game_rules)
            logging.info(f"Game '{game_name}' added to collection mapping")

        logging.info(f"Successfully ingested document: {file_path}")


if __name__ == "__main__":
    # Example usage of RAGIngestor to ingest a PDF document
    ingestor = RAGIngestor()

    # Specify the path to the PDF document
    file_path = "data/docs/tic_tac_toe.pdf"
    ingestor.ingest_document(file_path)
