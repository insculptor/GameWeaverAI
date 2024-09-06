"""
####################################################################################
#####                     File: src/rag/ingest_data.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####              Class to Ingest Data into from RAG Pipeline                 #####
####################################################################################
"""
import os

from src.rag.rag_pipeline_base import RAGPipeline


class RAGIngestor(RAGPipeline):
    """Class to handle document ingestion."""

    def ingest_document(self, file_path):
        """Ingests each section of the PDF as separate documents with embeddings into ChromaDB."""
        game_name = os.path.basename(file_path).replace(".pdf", "")
        game_id, game_rules = self.get_or_create_game_id(game_name)
        
        # Read sections from the PDF
        sections = self.read_pdf_sections(file_path)
        
        # Preprocess and split each section into chunks
        chunks = self.preprocess_and_split_sections(sections, chunk_size=300, chunk_overlap=50)
        
        # Create embeddings for the chunks
        embeddings = self.create_embeddings(chunks)
        
        # Ingest each chunk with metadata
        for section_name, chunk_list in chunks.items():
            full_section_text = sections[section_name]

            for i, chunk in enumerate(chunk_list):
                document_id = f"{game_name}_{section_name.replace(' ', '_')}_{i}"

                # Store both the full section text and individual chunk text
                document_metadata = {
                    "Game_ID": str(game_id),
                    "Game_Name": game_name,
                    "Section_name": section_name,
                    "Text": full_section_text,  # Complete section text
                    "Chunk_Text": chunk  # Chunk text
                }

                # Upsert the document into ChromaDB
                self.collection.upsert(
                    ids=[document_id],
                    documents=[chunk],
                    embeddings=[embeddings[section_name][i].tolist()],
                    metadatas=[document_metadata]
                )

        # Update the JSON file if it's a new game
        if not any(rule["Game Name"] == game_name for rule in game_rules):
            game_rules.append({"ID": game_id, "Game Name": game_name})
            self.write_collection_mapping(game_rules)
        
        print(f"Ingested document: {file_path}")
