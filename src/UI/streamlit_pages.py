"""
####################################################################################
#####                       File name: streamlit_pages.py                      #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/03/2024                              #####
#####                    Streamlit App Web Pages Helper File                   #####
####################################################################################
"""

import os

import PyPDF2
import streamlit as st
from dotenv import load_dotenv

#from src.rag.rag_pipeline import ingest_game_rules

# Load environment variables
load_dotenv()

# Path to the documents directory from .env
DOCS_PATH = os.getenv("DOCS_PATH")


VALID_GAME_DOCUMENT = """
<h3>The Valid Content of the Game File should have the following Sections:</h3>
<ol>
    <li><strong>Overview</strong>: A brief description of the game.</li>
    <li><strong>Game Setup</strong>: Details on how to set up the game.</li>
    <li><strong>How to Play</strong>: Instructions on playing the game.</li>
    <li><strong>Winning the Game</strong>: Conditions required to win the game.</li>
    <li><strong>Game Strategy</strong>: Tips and strategies for playing the game.</li>
    <li><strong>End of Game</strong>: How and when the game ends.</li>
</ol>
<p><em>Please ensure your document follows this structure for proper ingestion.</em></p>
"""

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
    st.title("Game Documents Admin")
    st.subheader("Upload Game Rule Documents")

    # Instructions for the admin
    st.write("""
    ### Instructions:
    - Please ensure the document follows the correct format as per the provided guidelines.
    - The document must include all game rules in a readable and structured manner.
    - **Example of the correct format**: The content should follow the same structure as `tic_tac_toe.pdf`.
    - The file should be in PDF format.
    """)

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
            ##ingest_game_rules(file_path)
        else:
            st.error("The uploaded file does not match the required format. Please check and upload again.")

