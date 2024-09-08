"""
####################################################################################
#####                     File: src/UI/streamlit_pages.py                      #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/03/2024                              #####
#####                    Streamlit App Web Pages Helper File                   #####
####################################################################################
"""

import os

import PyPDF2
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.controllers.executor import GameFlow
from src.rag.ingest_data import RAGIngestor
from src.rag.retrieve_data import RAGRetriever

# Load environment variables
load_dotenv()

# Path to the documents directory from .env
DOCS_PATH = os.getenv("DOCS_PATH")


# Load config.yaml file
config_path = os.path.join(os.getenv("ROOT_PATH"), "config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract section titles and descriptions from config.yaml
section_titles = config['game_rules']['section_titles']
section_descriptions = config['game_rules']['section_descriptions']

# Dynamically create the HTML document
VALID_GAME_DOCUMENT = """
<h3><strong>Instructions</strong>: The Valid Content of the Game File should have the following Sections:</h3>
<ol>
"""

for section in section_titles:
    description = section_descriptions.get(section, "Description not available.")
    VALID_GAME_DOCUMENT += f"    <li><strong>{section}</strong>: {description}</li>\n"

VALID_GAME_DOCUMENT += """
</ol>
<p><em>Please ensure your document follows this structure for proper ingestion.</em></p>
"""

# Print the dynamically generated VALID_GAME_DOCUMENT
print(VALID_GAME_DOCUMENT)


REQUIRED_SECTIONS = [
    "Overview",
    "Game Setup",
    "How to Play",
    "Winning the Game",
    "Game Strategy",
    "End of Game"
]

def validate_pdf(file):
    """Validates if the uploaded PDF contains all the required sections."""
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Check for the presence of all required sections
    missing_sections = [section for section in REQUIRED_SECTIONS if section not in text]

    if missing_sections:
        st.error(f"The following required sections are missing from the document: {', '.join(missing_sections)}")
        return False

    # All sections are present
    return True

def game_documents_admin():
    """Game Documents Admin page for uploading and validating game rule documents."""
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("Gameweaver Admin - Upload Game Rule Documents", divider="rainbow")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Validate the file content
        if validate_pdf(uploaded_file):
            # Save the uploaded file to the specified path in .env
            os.makedirs(DOCS_PATH, exist_ok=True)
            file_path = os.path.join(DOCS_PATH, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            st.success(f"File uploaded successfully: {file_path}")

            # Ingest the uploaded file into the vector database
            ingestor = RAGIngestor()
            ingestor.ingest_document(file_path)
        else:
            st.error("The uploaded file does not match the required format. Please check and upload again.")

    st.markdown(VALID_GAME_DOCUMENT, unsafe_allow_html=True)

def metadata_viewer():
    """Page for viewing metadata from the vector database based on a game ID."""
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("Gameweaver Admin - Metadata Viewer", divider="rainbow")
    

    game_id = st.text_input("Enter Game ID:")
    if st.button("Fetch Metadata"):
        if game_id:
            retriever = RAGRetriever()
            metadata = retriever.fetch_document_metadata(game_id)

            if metadata:
                st.write(f"Metadata for Game ID {game_id}:")
                st.json(metadata)
            else:
                st.error(f"No metadata found for Game ID: {game_id}")
        else:
            st.error("Please enter a valid Game ID.")
            
## Home Page
def generate_game_page():
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("Generate and Play Your Game with AI", divider="rainbow")

    game_choice = st.text_input("Enter the name of the game you would like to play:")
    player_mode = st.radio("Choose mode:", ("Single Player", "Multiplayer"))

    # Add a Play button
    if st.button("Play"):
        if game_choice:
            st.info(f"Processing your request for '{game_choice}'...")

            # Initialize GameFlow to handle game generation
            game_flow = GameFlow()

            # Execute the GameFlow engine to get the game rules and code
            game_code = game_flow.play_game(game_choice)

            if game_code:
                st.success(f"Game '{game_choice}' generated successfully!")
                
                # Display the rules first
                game_metadata = game_flow.retriever.fetch_document_metadata_by_name(game_choice)
                if game_metadata:
                    st.subheader(f"Game Rules for {game_choice}:")
                    for section, content in game_metadata.items():
                        if section in ["ID", "Game_Name"]:
                            continue
                        st.markdown(f"**{section}**: {content.get('Text', 'No content available')}")

                # Execute the Python code to render the game (this will vary depending on the type of game generated)
                st.subheader(f"Play {game_choice}:")

                # Create a placeholder for the game UI
                game_placeholder = st.empty()

                # Play the game depending on the mode (Single Player or Multiplayer)
                if player_mode == "Single Player":
                    exec(game_code)  # Executing the Python code generated by LLM for Single Player mode
                    # TODO: Implement game logic where AI plays against the user
                else:
                    exec(game_code)  # Executing the Python code generated by LLM for Multiplayer mode
                    # TODO: Implement game logic where two players can play

            else:
                st.error(f"Failed to generate the game for '{game_choice}'. Please try again.")

        else:
            st.error("Please enter a game name.")