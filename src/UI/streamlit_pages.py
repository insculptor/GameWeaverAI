"""
####################################################################################
#####                     File: src/UI/streamlit_pages.py                      #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/03/2024                              #####
#####                    Streamlit App Web Pages Helper File                   #####
####################################################################################
"""

import os
import subprocess

import PyPDF2
import streamlit as st
import yaml
from dotenv import load_dotenv

from src.controllers.executor import GameFlow
from src.rag.ingest_data import RAGIngestor
from src.rag.retrieve_data import RAGRetriever
from src.UI.htmltemplates import css

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
    # Add the custom CSS for styling
    st.markdown(css, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("Generate and Play Your Game with AI", divider="rainbow")

    game_choice = st.text_input("Enter the name of the game you would like to play:")

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
                        # Apply CSS style to section names using strong tag
                        st.markdown(f"<strong>{section}</strong>: {content.get('Text', 'No content available')}", unsafe_allow_html=True)

                # Render a placeholder for the game being launched
                st.subheader(f"Launching {game_choice} in terminal...")
                st.info("A new terminal window will open where you can play the game.")

                # Create a Python file with the game code
                game_file = f"{game_choice.lower().replace(' ', '_')}_game.py"
                with open(game_file, 'w') as f:
                    f.write(game_code)

                # Launch the game in a new terminal window using subprocess
                launch_game_in_terminal(game_file)

            else:
                st.error(f"Failed to generate the game for '{game_choice}'. Please try again.")

        else:
            st.error("Please enter a game name.")

def launch_game_in_terminal(game_file):
    """
    Launch the Python game in a new terminal window.
    """
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(['start', 'cmd', '/k', f'python {game_file}'], shell=True)
        else:  # macOS/Linux
            subprocess.Popen(['gnome-terminal', '--', 'python3', game_file], shell=False)
    except Exception as e:
        st.error(f"Failed to launch the game in the terminal: {e}")