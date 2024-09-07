"""
####################################################################################
#####                     File: src/controllers/executor.py                    #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####      Execution control logic for generating code or game rules           #####
####################################################################################
"""

from src.controllers.prompts import generate_code_prompt, generate_game_rules_prompt
from src.models.openai import generate_code_openai, generate_rules_openai
from src.models.vllm_server import generate_code, generate_rules


def execute_code_generation(game_metadata):
    """
    Control the execution for generating code based on existing game metadata.
    If FastAPI LLM is down, fallback to OpenAI.

    Args:
        game_metadata (dict): Dictionary containing game metadata (overview, setup, etc.)

    Returns:
        str: The generated Python code for the game.
    """
    try:
        # Generate prompt to create code based on game metadata
        prompt = generate_code_prompt(game_metadata)
        
        # Try calling the FastAPI LLM service
        generated_code = generate_code(prompt)
        return generated_code
    except Exception as e:
        print(f"FastAPI LLM server failed: {e}. Falling back to OpenAI.")
        
        # If FastAPI fails, fallback to OpenAI
        return generate_code_openai(prompt)


def execute_rules_generation(game_name):
    """
    Control the execution for generating game rules when the game doesn't exist in the vector database.
    If FastAPI LLM is down, fallback to OpenAI.

    Args:
        game_name (str): The name of the game the user wants to create.

    Returns:
        str: The generated game rules.
    """
    try:
        # Generate a creative prompt for game rules
        prompt = generate_game_rules_prompt(game_name)
        
        # Try calling the FastAPI LLM service for rules generation
        generated_rules = generate_rules(prompt)
        return generated_rules
    except Exception as e:
        print(f"FastAPI LLM server failed: {e}. Falling back to OpenAI.")
        
        # If FastAPI fails, fallback to OpenAI
        return generate_rules_openai(prompt)
