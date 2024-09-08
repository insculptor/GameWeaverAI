"""
####################################################################################
#####                     File: src/controllers/executor.py                    #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/07/2024                              #####
#####             Execution control logic for Generating Gameplay              #####
####################################################################################
"""

# from src.controllers.prompts import generate_code_prompt, generate_game_rules_prompt
# from src.rag.ingest_data import RAGIngestor
# from src.rag.retrieve_data import RAGRetriever


# Generate Gameplay

import os

from dotenv import load_dotenv

load_dotenv()


def list_files_and_dirs_recursive(directory, directories_to_exclude=[]):
    """
    Recursively lists all files and directories inside the given directory, excluding specified directories.

    Args:
        directory (str): The path of the directory to list.
        directories_to_exclude (list): A list of directory names to exclude from traversal.

    Returns:
        list: A list of file and directory paths inside the given directory.
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        # Modify dirs in-place to exclude specified directories
        dirs[:] = [d for d in dirs if d not in directories_to_exclude]
        
        for name in dirs:
            file_list.append(os.path.join(root, name))  # Add directories to the list
        for name in files:
            file_list.append(os.path.join(root, name))  # Add files to the list

    return file_list

# Example usage
directory_path = os.getenv("ROOT_PATH")  # Replace with the actual path
exclude_dirs = ['__pycache__', '.venv','dunzhang','16bae58a-783c-43bb-8018-d4b0271a59bf',".git"]  # Directories to exclude

files_and_dirs = list_files_and_dirs_recursive(directory_path, exclude_dirs)
for item in files_and_dirs:
    print(item)
