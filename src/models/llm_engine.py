"""
####################################################################################
#####                    File: src/models/llm.py                               #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####     Unified LLM Service Class for JarvisLabs (primary) and OpenAI        #####
####################################################################################
"""

import logging
import os
import re
import sys

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
ROOT_PATH = os.environ.get("ROOT_PATH")
sys.path.append(ROOT_PATH)
print(f"[INFO]: {ROOT_PATH=}")
config_path = os.path.join(ROOT_PATH, 'config.yaml')
with open(config_path, 'r') as f:
        config =  yaml.safe_load(f)

from src.controllers.prompts import generate_code_prompt, generate_game_rules_prompt


# Setting up logging based on the config.yaml file
class LLMService:
    """
    LLMService class that handles code and game rules generation using either JarvisLabs or OpenAI.
    JarvisLabs is the primary service, and OpenAI is used as a fallback.
    """
    def __init__(self):
        """
        Initialize the LLMService class by loading the necessary environment variables and config settings.
        """
        # Load environment variables
        load_dotenv()

        # Load config.yaml for model names and logging level
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set logging level from config.yaml
        log_level = self.config.get("logging", {}).get("DEBUG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),  # Default to INFO if not found
            format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logging.info(f"Logging level set to {log_level}")

        # JarvisLabs API details
        self.jarvis_api_key = os.getenv("JARVIS_API_KEY")
        self.jarvis_code_endpoint = os.getenv("JARVIS_OLLAMA_CODE_ENDPOINT")

        # OpenAI configuration
        self.client = OpenAI()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = self.config['models']['OPENAI_LLM_MODEL']

        # JarvisLabs - Ollama Models
        self.jarvis_code_llm = self.config['models']['JARVIS_CODE_LLM']
        self.jarvis_rules_llm = self.config['models']['JARVIS_RULES_LLM']
        self.jarvis_client = OpenAI(
                    base_url=self.jarvis_code_endpoint,
                    api_key=self.jarvis_api_key)
        
        # Game Rules Configuration
        self.section_titles = self.config['game_rules']['section_titles']
        self.section_descriptions = self.config['game_rules']['section_descriptions']

        logging.info("LLMService initialized with models from config.yaml.")

    def is_jarvis_available(self):
        """
        Check if the JarvisLabs API is available by making a simple request.

        Returns:
            bool: True if JarvisLabs is available, False otherwise.
        """
        try:
            print(f"Checking JarvisLabs availability at {self.jarvis_code_endpoint}...")
            client = OpenAI(base_url=self.jarvis_code_endpoint, api_key=self.jarvis_api_key)
            completion = client.chat.completions.create(model=self.jarvis_rules_llm,
                                                        messages=[{"role": "user", "content": "Hi"}])
            
            if completion:
                print(f"Response from JarvisLabs: {completion.choices[0].message.content}")
                print("JarvisLabs is available.")
                return True
            else:
                print("[JARVIS][UNAVAILABLE]: JarvisLabs is not available. Please check the API key and endpoint.")
        except requests.RequestException:
            # Catching any requests-related errors, logging them, and returning False
            print("JarvisLabs is unavailable.")
            return False
        except Exception:
            # Catching all other general exceptions
            print("An unexpected error occurred while checking JarvisLabs availability.")
            return False

    def openai_generate(self, prompt: str):
        """
        Generates LLM response using Open AI for any Prompt.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code or an error message.
        """
        logging.info("Generating code using OpenAI...")
        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": prompt}]
            )
            llm_response = completion.choices[0].message.content
            logging.info("Response generated successfully using OpenAI.")
            return llm_response.strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return f"OpenAI API error: {e}"
    
    def generate_code_jarvis(self, prompt: str):
        """
        Generates code using JarvisLabs.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code or an error message.
        """
        logging.info(f"Generating code using {self.jarvis_code_llm} hosted at Jarvis...")
        try:
            completion = self.jarvis_client.chat.completions.create(
                        model=self.jarvis_code_llm,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                        )

            llm_response = completion.choices[0].message.content
            logging.info(f"Code generated successfully using {self.jarvis_code_llm} hosted at Jarvis.")
            return self.extract_python_code(llm_response.strip())
        except Exception as e:
            logging.error(f"Jarvis API error: {e}")
            return f"Jarvis API error: {e}"

    def extract_python_code(self, response: str) -> str:
        """
        Extract Python code from the LLM response by identifying code blocks between '''python and '''.

        Args:
            response (str): The LLM response that may contain Python code.

        Returns:
            str: Extracted Python code, or the full response if no code block is found.
        """
        logging.info("Extracting Python code from LLM response...")
        code_match = re.search(r"```python(.*?)```", response, re.DOTALL)
        if code_match:
            logging.info("Python code extracted successfully.")
            return code_match.group(1).strip()
        else:
            logging.warning("No Python code found in the response. Returning full response.")
            return response

    def generate_code(self, prompt: str):
        """
        Unified function for generating code. Tries JarvisLabs first, falls back to OpenAI if Jarvis is unavailable.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code.
        """
        logging.info("Attempting to generate code with JarvisLabs...")
        if self.is_jarvis_available():
            return self.generate_code_jarvis(prompt)
        else:
            logging.info("Falling back to OpenAI for code generation.")
            return self.extract_python_code(self.openai_generate(prompt).strip()) 

    ############################
    ### Game Rules Generation ##
    ############################
    
    def generate_rules_jarvis(self, prompt: str):
        """
        Generates game rules using JarvisLabs.

        Args:
            prompt (str): The input prompt to generate rules.

        Returns:
            str: The generated game rules or an error message.
        """
        logging.info(f"Generating game rules using  {self.jarvis_rules_llm} hosted at JarvisLabs...")
        try:
            completion = self.jarvis_client.chat.completions.create(
                        model=self.jarvis_rules_llm,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                        )

            llm_response = completion.choices[0].message.content
            logging.info(f"Game Rules generated successfully using {self.jarvis_rules_llm} hosted at Jarvis.")
            return llm_response.strip()
        except Exception as e:
            logging.error(f"Jarvis API error: {e}")
            return f"Jarvis API error: {e}"

    def generate_rules(self, prompt: str):
        """
        Unified function for generating game rules. Tries JarvisLabs first, falls back to OpenAI if Jarvis is unavailable.

        Args:
            prompt (str): The input prompt to generate rules.

        Returns:
            str: The generated game rules.
        """
        logging.info("Attempting to generate game rules with JarvisLabs...")
        if self.is_jarvis_available():
            return self.generate_rules_jarvis(prompt)
        else:
            logging.info("Falling back to OpenAI for rules generation.")
            return self.openai_generate(prompt)  # OpenAI fallback
        
    def parse_game_rules_to_metadata(self, game_rules_text:str):
        """
        Parses the generated game rules text and converts it into the metadata format.

        Args:
            game_rules_text (str): The generated game rules text.

        Returns:
            dict: The metadata dictionary containing each section with its respective text.
        """
        # Initialize the metadata dictionary using section titles from the config
        metadata = {section: {"Text": "", "Description": self.section_descriptions.get(section, "")} for section in self.section_titles}

        # Pattern to match the sections in the generated text
        section_pattern = re.compile(r"####\s*(\d\.\s*)?(" + "|".join(re.escape(title) for title in self.section_titles) + ")", re.IGNORECASE)

        # Split the text based on section headings
        split_text = section_pattern.split(game_rules_text)

        # Populate the metadata dictionary
        current_section = None
        for i in range(1, len(split_text), 3):
            section_name = split_text[i+1].strip()  # Strip out any whitespace
            section_content = split_text[i+2].strip()  # Get the section's content
            if section_name in metadata:
                metadata[section_name]["Text"] = section_content

        return metadata


# Example usage
if __name__ == "__main__":
    llm_service = LLMService()
    print("*"*100)
    print("[INFO]: Testing LLM Generation for Code and Game Rules")
    game_rules_prompt = generate_game_rules_prompt("Bill Gates")
    print("Game Rules Prompt:\n", game_rules_prompt)
    generated_rules = llm_service.generate_rules(game_rules_prompt)
    print("Generated Game Rules:\n", generated_rules)
    print("*"*100)
    print("[INFO]: Parsing Game Rules to Metadata")
    metadata = llm_service.parse_game_rules_to_metadata(generated_rules)
    print("*"*100)
    print(f"[INFO]: Generated Metadata:\n {metadata}")
    code_prompt = generate_code_prompt(metadata)
    print("*"*100)
    print("Code Generation Prompt:\n", code_prompt)
    generated_code = llm_service.generate_code(code_prompt)
    print("*"*100)
    print("Generated Code:\n", generated_code)