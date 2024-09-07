"""
####################################################################################
#####                     File: src/controller/prompt.py                       #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####            Prompt generation for code and game rules                     #####
####################################################################################
"""

import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for verbose output
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

    # Extract the required sections from the metadata
    overview = game_metadata.get('Overview', {}).get('Text', 'Not available')
    game_setup = game_metadata.get('Game Setup', {}).get('Text', 'Not available')
    how_to_play = game_metadata.get('How to Play', {}).get('Text', 'Not available')
    winning_the_game = game_metadata.get('Winning the Game', {}).get('Text', 'Not available')
    game_strategy = game_metadata.get('Game Strategy', {}).get('Text', 'Not available')
    end_of_game = game_metadata.get('End of Game', {}).get('Text', 'Not available')

    # Log the extracted metadata
    logging.debug(f"Overview: {overview[:100]}...")  # Log first 100 characters for brevity
    logging.debug(f"Game Setup: {game_setup[:100]}...")
    logging.debug(f"How to Play: {how_to_play[:100]}...")
    logging.debug(f"Winning the Game: {winning_the_game[:100]}...")
    logging.debug(f"Game Strategy: {game_strategy[:100]}...")
    logging.debug(f"End of Game: {end_of_game[:100]}...")

    # Create the prompt based on the extracted sections
    prompt = f"""
    You are a Python expert and you are tasked with generating the code for a game. Here are the components of the game:

    Overview: {overview}
    Game Setup: {game_setup}
    How to Play: {how_to_play}
    Winning the Game: {winning_the_game}
    Game Strategy: {game_strategy}
    End of Game: {end_of_game}

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

    # Create the prompt to generate game rules
    prompt = f"""
    You are a creative game designer and Python programmer. You need to design a new game called "{game_name}".
    
    Please create detailed game rules for this new game. Include the following sections:

    1. Overview: A brief description of the game.
    2. Game Setup: Details on how to set up the game.
    3. How to Play: Instructions on how to play the game.
    4. Winning the Game: Conditions required to win the game.
    5. Game Strategy: Tips and strategies for playing the game.
    6. End of Game: How and when the game ends.

    Once the rules are established, ensure the rules can be used to generate Python code to play this game. Be creative and come up with a fun, interactive game idea.
    """

    # Log completion
    logging.info(f"Game rules prompt generated successfully for {game_name}.")
    
    return prompt
