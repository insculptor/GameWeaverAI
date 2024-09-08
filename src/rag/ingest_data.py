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
import re
import sys

import yaml
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH","/app")
sys.path.append(ROOT_PATH)
print(f"[INFO]: {ROOT_PATH=}")
config_path = os.path.join(ROOT_PATH, 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

from src.rag.rag_pipeline_base import RAGPipeline

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to see verbose output
    format="%(filename)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to the console
)

class RAGIngestor(RAGPipeline):
    """Class to handle document ingestion into ChromaDB."""

    def ingest_game_rules(self, game_name, game_rules):
        """
        Ingests generated game rules into ChromaDB.
        
        Args:
            game_name (str): The name of the game.
            game_rules (str): The generated game rules text.
        """
        logging.info(f"Starting ingestion for game: {game_name}")

        game_id, game_rules_list = self.get_or_create_game_id(game_name)
        logging.debug(f"Game ID: {game_id}, Game Name: {game_name}")

        # Preprocess and split the game rules into sections
        sections = self.preprocess_generated_rules(game_rules)
        logging.info(f"Sections found in game rules: {list(sections.keys())}")

        # Preprocess and split each section into chunks
        chunks = self.preprocess_and_split_sections(sections, chunk_size=300, chunk_overlap=50)
        if not chunks:
            logging.error(f"No chunks created for game: {game_name}")
            return

        # Create embeddings for the chunks
        embeddings = self.create_embeddings(chunks)
        logging.debug("Embeddings generated for game rules chunks")

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
        if not any(rule["Game Name"] == game_name for rule in game_rules_list):
            game_rules_list.append({"ID": game_id, "Game Name": game_name})
            self.write_collection_mapping(game_rules_list)
            logging.info(f"Game '{game_name}' added to collection mapping")

        logging.info(f"Successfully ingested game rules for '{game_name}'")

    def preprocess_generated_rules(self, game_rules_text):
        """
        Preprocess the generated game rules text into sections and match them with the expected section titles.

        Args:
            game_rules_text (str): The full text of the generated game rules.

        Returns:
            dict: A dictionary where keys are section titles from config.yaml and values are the corresponding text.
        """
        sections = {}
        current_section = None

        # Get the section titles from config.yaml
        section_titles = config['game_rules']['section_titles']
        logging.debug(f"Section titles from config: {section_titles}")

        # Prepare to clean and normalize section titles
        def normalize_section_title(title):
            logging.debug(f"Normalizing section title: {title}")
            # Normalize: remove spaces, lowercase, but maintain them with spaces, not concatenated
            return re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().strip()

        # Preprocess and normalize the config section titles (preserve spaces in titles)
        normalized_section_titles = {normalize_section_title(title): title for title in section_titles}
        logging.debug(f"Normalized section titles: {normalized_section_titles}")

        # Clean and prepare game rules text (convert to lowercase, clean extra characters)
        cleaned_rules = self.clean_text(game_rules_text.lower())
        logging.debug(f"Cleaned game rules text: {cleaned_rules}")

        # Split the game rules text into lines
        lines = cleaned_rules.splitlines()
        
        for line in lines:
            # Check if a section title matches
            for title_key in normalized_section_titles:
                if re.search(rf'\b{title_key}\b', line):  # Check for section title in the line
                    current_section = normalized_section_titles[title_key]
                    sections[current_section] = []
                    logging.debug(f"Matched section: {current_section}")
                    break
            
            if current_section:
                # Accumulate lines under the current section
                sections[current_section].append(line)

        # Join lines in each section
        for section in sections:
            sections[section] = " ".join(sections[section]).strip()

        logging.debug(f"Final sections: {sections}")
        return sections

    def clean_text(self, text):
        """
        Cleans the input text by removing unnecessary formatting like '###', '**', etc.

        Args:
            text (str): The raw text to be cleaned.

        Returns:
            str: The cleaned text.
        """
        # Remove Markdown-like formatting (remove ###, **, ##, and extra spaces)
        logging.debug(f"Cleaning text: {text}")        
        text = re.sub(r'[#*]+', '', text)  # Removes #, *, **, ###
        text = text.replace(':', '')
        text = re.sub(r'^\s*\d+\.\s*', '', text)
        print(f"Cleaned text: {text}")
        logging.debug(f"Cleaned text: {text}")
        return text.strip()

    def ingest_document(self, file_path):
        """
        Ingests a document (like PDF) into ChromaDB from UI.

        Args:
            file_path (str): Path to the PDF file to be ingested.
        """
        logging.info(f"Starting ingestion for document: {file_path}")

        # Extract the game name from the file path (assumes PDF file format)
        game_name = os.path.basename(file_path).replace(".pdf", "")
        game_id, game_rules = self.get_or_create_game_id(game_name)
        print("*"*50)
        print(game_rules)
        print("*"*50)

        logging.debug(f"Game ID: {game_id}, Game Name: {game_name}")

        # Read sections from the PDF
        sections = self.read_pdf_sections(file_path)
        sections = self.clean_text(sections)
        if not sections:
            logging.error(f"No sections found in {file_path}")
            return

        logging.info(f"Sections found: {list(sections)}")

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
    # Example usage of RAGIngestor to ingest game rules
    ingestor = RAGIngestor()

    # Specify the game name and generated rules
    game_name = "F.R.I.E.N.D.S"
    game_rules_text = """
    **F.R.I.E.N.D.S Game Rules**
### 1. Overview:
"F.R.I.E.N.D.S" is a social deduction board game where players take on the roles of quirky friends trying to outmaneuver each other in building the most efficient and fun-filled day together. The game encourages strategic thinking, negotiation, and bluffing as players manage their time, resources, and relationships with one another.
Players will navigate through a series of interactions, tasks, and misadventures that test their ability to collaborate, communicate, and deceive each other. The ultimate goal is to build the best day together while sabotaging your friends' attempts to do the same.
**Game Components:**
* Game board featuring a modular map of various locations and activities
* 6 Character cards with unique abilities and strengths
* 100 Event cards that drive the gameplay and interactions between players
* Resource tokens for managing time, friendship points, and cash
* Friend Meter track to monitor relationships and trust among friends
* Scorepad for keeping track of individual player scores
### 2. Game Objective:
**The Game Objective is to earn the highest score by building a satisfying day with your fellow friends while sabotaging their attempts to do the same.**
There are three ways to earn points:
1. **Friendship Points:** By successfully completing shared tasks or resolving conflicts between players.
2. **Cash Earned:** Through various entrepreneurial activities or selling goods and services among friends.
3. **Bad Memory Points (BM):** Through negative interactions, failed tasks, and disputes.
The player with the highest total score at the end of the game wins! In case of a tie, the player with the most Friendship Points takes precedence over the others.
### 3. Game Setup:
1. **Initial Preparation:** Shuffle the Character cards and distribute them randomly among the players.
2. **Location Board Assembly:** Each location on the board represents different settings or activities that will appear in the game. Players roll a die to determine which player goes first, and they place their Character card on any empty space on the board for their turn sequence.
**Player Roles:**
Each player is introduced as having unique strengths, fears, hobbies, and relationship dynamics (Friend Meter). This aspect adds depth and encourages players to develop character-driven strategies in achieving a harmonious friendship or cunning exploitation of others' flaws!
### 4. How to Play:
- **Turn Sequence:** Based on the Character card rolled for turn order during setup, players will take turns.
- **Gameplay Loop:**
1. Choose an Area Card from the deck (up to two locations can be accessed).
2. Resolve any actions, conversations, or tasks prompted by these card locations.
3. Earn rewards (Resources and/or Friendship Points), incur fines, gain insights into others' intentions or conflicts.
4. Move time counters on individual Character cards, adjusting player turns accordingly.
Players interact based on the Event cards that emerge during this interaction phase:
- Attempt to build friendships with another character for better shared tasks' success in resource optimization (Friend Meter adjustments).
- Misuse their unique capabilities and sabotage other players' efforts by applying negative influence cards.
- Participate in time-sharing activities, such as "Help," where resources are pooled when mutually beneficial.
- Pursue personal objectives (cash earnings or skill-building events), using Event Cards strategically to disrupt the day of another character while gaining benefits for themselves.
Players take turns throughout until all locations on a card deck have been accessed.
### 5. Winning the Game:
The game ends suddenly whenever, as it has no predetermined length with various factors altering the possible end-point:
1. Resource depletion.
2. Maximum time reached on the last remaining Player Cards.
3. Last player reaching a cash total exceeding all other Player cards' combined values
### 6. Game Strategy and Tips for Playing:
**Master Strategic Bluffing:** Use your Character abilities, knowing how they affect interpersonal dynamics, personal growth opportunities, or financial investments to gain leverage in strategic trades without raising anyone else's suspicions.
Negotiate the delicate balance between friendly relationships (Friend Meter) and exploiting weaknesses found in opponents’ vulnerabilities.
Consider your Character’s unique traits when navigating tasks. You may want to use them strategically to build an event of the most efficiency while possibly manipulating another player as 'a partner', even if only temporarily, for furthering personal interest.
**Build a well-orchestrated campaign through resource sharing to achieve long-term objectives, all the time being aware of resource fluctuations that may threaten your plan.**
Be mindful and prepare for every turn’s uncertainties since some events can trigger conflicts while others support a harmonious friendship!
Players need to be creative in navigating a dynamic environment. This game requires social deduction skills because friends may deceive each other when advancing their goals while managing the fragile relationships between them.
### 7. End of Game:
As per defined end conditions, at least one player will emerge as the winner with more total points accumulated across various metrics (Friendship Points and Cash).
    """
    print(f"[INFO]: Ingesting game rules for '{game_name}'...")
    ingestor.ingest_game_rules(game_name, game_rules_text)
