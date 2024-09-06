"""
####################################################################################
#####                    File: src/UI/streamlit_app.py                         #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/03/2024                              #####
#####                Streamlit Application UI Main File                        #####
####################################################################################
"""

import os
import sys

# Add the Application root to Python path and set it as an environment variable
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT_PATH)
os.environ["ROOT_PATH"] = ROOT_PATH  # Add ROOT_PATH to the environment
print(f"Added {ROOT_PATH} to the Python path and environment.")

import streamlit as st
from dotenv import load_dotenv

from src.UI.htmltemplates import css
from src.UI.streamlit_pages import (
    game_documents_admin,
    generate_game_page,
    metadata_viewer,
)

# Load environment variables
load_dotenv()

# Path to the documents directory from .env
DOCS_PATH = os.getenv("DOCS_PATH")

# Page configuration and styling
st.set_page_config(page_title="GameWeaverAI", page_icon="🤖", layout="wide",initial_sidebar_state="collapsed")
st.markdown(css, unsafe_allow_html=True)

def main():
    st.sidebar.title("GameWeaverAI - Navigation")
    pages = ["GameweaverAI -  Home","Admin - Upload Game Document", "Admin - Game Data Viewer"]
    choice = st.sidebar.radio("Select a Page", pages)

    if choice == "GameweaverAI -  Home":
        generate_game_page()
    elif choice == "Admin - Upload Game Document":
        game_documents_admin()
    elif choice == "Admin - Game Data Viewer":
        metadata_viewer()

if __name__ == "__main__":
    main()
