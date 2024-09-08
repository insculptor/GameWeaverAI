"""
####################################################################################
#####                     File: src/controller/prompt.py                       #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####                Prompt generation for code and game rules                 #####
####################################################################################
"""

import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.getenv("ROOT_PATH","/app")
print(f"[INFO]: {ROOT_PATH=}")
config_path = os.path.join(ROOT_PATH, 'config.yaml')
with open(config_path, 'r') as f:
        config =  yaml.safe_load(f)
        

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for verbose output
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[logging.StreamHandler()]  # Logs to the console
)

def generate_code_prompt(game_metadata):
    """
    Generate a prompt for the LLM to generate code based on existing game rules from the vector database.

    Args:
        game_metadata (dict): Dictionary containing the sections of the game (overview, setup, etc.).

    Returns:
        str: The prompt to generate code for the game.
    """
    # Log the start of prompt generation
    logging.info("Starting code prompt generation...")

    # Load config for section titles
    section_titles = config['game_rules']['section_titles']

    # Create a dictionary to keep track of sections already added to the prompt
    added_sections = set()

    # Initialize the prompt text
    prompt = """
    You are a Python expert and you are tasked with generating the code for a game. Here are the components of the game:
    """
    
    # Dynamically generate the prompt based on the sections from the config
    for section in section_titles:
        # Normalize section name to handle case sensitivity
        normalized_section = section.lower().strip()

        if normalized_section not in added_sections:
            # Ensure the section is not repeated
            section_data = game_metadata.get(section, {}).get('Text', 'Not available')
            prompt += f"\n    {section}: {section_data}"
            added_sections.add(normalized_section)  # Mark this section as added

    prompt += """
    
    Based on the above information, generate the Python code for this game. Make sure the game is functional and can be played in a terminal or web interface.
    """

    # Log completion
    logging.info("Code prompt generated successfully.")
    
    return prompt


def generate_game_rules_prompt(game_name):
    """
    Generate a creative prompt for the LLM to generate game rules for a new game.

    Args:
        game_name (str): The name of the game that the user wants to create.

    Returns:
        str: The prompt to generate game rules for a new game.
    """
    # Log the start of prompt generation
    logging.info(f"Starting game rules prompt generation for {game_name}...")

    # Load config for section titles and descriptions
    section_titles = config['game_rules']['section_titles']
    section_descriptions = config['game_rules']['section_descriptions']

    # Create the prompt dynamically based on sections
    prompt = f"""
    You are a creative game designer and Python programmer. You need to design a new game called "{game_name}".
    
    Please create detailed game rules for this new game. Include the following sections:
    """
    
    for idx, section in enumerate(section_titles, 1):
        description = section_descriptions.get(section, "No description available.")
        prompt += f"\n    {idx}. {section}: {description}"

    prompt += """
    
    Once the rules are established, ensure the rules can be used to generate Python code to play this game. Do not generate Python Code, only the rules based on above sections. Be creative and come up with a fun, interactive game idea.
    """

    # Log completion
    logging.info(f"Game rules prompt generated successfully for {game_name}.")
    
    return prompt


# Example usage
if __name__ == "__main__":
    # Example metadata to generate a code prompt
    example_metadata = {
        "Overview": {"Text": "This is a simple Tic-Tac-Toe game."},
        "Game Objective": {"Text": "The objective is to get 3 in a row."},
        "Game Setup": {"Text": "The game is played on a 3x3 grid."},
        "How to Play": {"Text": "Players take turns to place their symbols."},
        "Winning the Game": {"Text": "The first player to get 3 in a row wins."},
        "Game Strategy": {"Text": "Block your opponent's moves."},
        "End of Game": {"Text": "The game ends when all squares are filled."}
    }
    
    # Generate code prompt
    code_prompt = generate_code_prompt(example_metadata)
    print("Generated Code Prompt:\n", code_prompt)

    # Generate game rules prompt
    game_rules_prompt = generate_game_rules_prompt("Space Adventure")
    print("Generated Game Rules Prompt:\n", game_rules_prompt)
