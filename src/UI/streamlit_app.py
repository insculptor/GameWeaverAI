"""
####################################################################################
#####                    File name: streamlit_app.py                           #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/03/2024                              #####
#####                Streamlit Application UI Main File                        #####
####################################################################################
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv
from streamlit_pages import VALID_GAME_DOCUMENT

# Add the src directory to the Python path
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_PATH)
print(f"Added {ROOT_PATH} to the Python path.")

from src.rag.rag_pipeline import (
    fetch_game_rules,
    ingest_game_rules,
)
from src.UI.htmltemplates import css

# Load environment variables
load_dotenv()

# Path to the documents directory from .env
DOCS_PATH = os.getenv("DOCS_PATH")

# Page configuration and styling
st.set_page_config(page_title="GameWeaverAI", page_icon="🤖", layout="wide",initial_sidebar_state="collapsed")
st.markdown(css, unsafe_allow_html=True)

def main():
    st.sidebar.title("GameWeaverAI - Navigation")
    pages = ["GameWeaverAI - Home", "GameWeaverAI - Adminpanel"]
    choice = st.sidebar.radio("", pages)

    if choice == "GameWeaverAI - Home":
        generate_game_page()
    elif choice == "GameWeaverAI - Adminpanel":
        admin_upload_page()

def generate_game_page():
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("Generate and Play Your Game with AI", divider="rainbow")

    game_choice = st.text_input("Enter the name of the game you would like to play:")
    player_mode = st.radio("Choose mode:", ("Single Player", "Multiplayer"))

    if st.button("Generate Game"):
        if game_choice:
            # Fetch game rules from the RAG pipeline
            game_rules = fetch_game_rules(game_choice)
            if game_rules:
                st.success(f"Found existing rules for {game_choice}. Generating the game...")
                # TODO: Pass the rules to LLM Code Generator (to be implemented)
                st.write(f"Game Rules: {game_rules}")  # Display fetched rules for now
            else:
                st.warning(f"No existing rules found for {game_choice}. Generating new rules...")
                # TODO: Call LLM Rules Generator to create new rules (to be implemented)
        else:
            st.error("Please enter a game name.")

def admin_upload_page():
    st.markdown("<h1 style='text-align: center;'>🤖 GameWeaverAI </h1>", unsafe_allow_html=True)
    st.header("GameWeaverAI - Admin Panel", divider="rainbow")
    st.subheader("Upload a PDF document to ingest into the database.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to the specified path in .env
        os.makedirs(DOCS_PATH, exist_ok=True)
        file_path = os.path.join(DOCS_PATH, uploaded_file.name)

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File uploaded successfully: {file_path}")

        # Ingest the uploaded file into the vector database
        ingest_game_rules(file_path)
        
    st.markdown(VALID_GAME_DOCUMENT, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
