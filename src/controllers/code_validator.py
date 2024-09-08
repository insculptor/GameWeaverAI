"""
####################################################################################
#####                File: src/controllers/code_validator.py                   #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/07/2024                              #####
#####               Class to Manage Game Flow for GameWeaverAI                 #####
####################################################################################
"""
import ast
import logging
import os
import sys
import traceback

import yaml
from dotenv import load_dotenv

load_dotenv()
ROOT_PATH = os.environ.get("ROOT_PATH")
sys.path.append(ROOT_PATH)
print(f"[INFO]: {ROOT_PATH=}")
config_path = os.path.join(ROOT_PATH, 'config.yaml')
with open(config_path, 'r') as f:
        config =  yaml.safe_load(f)
        

# Setting up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(filename)s - %(message)s",
    handlers=[logging.StreamHandler()]
)


class CodeValidator:
    """
    CodeValidator class to validate and check if the provided Python code is syntactically valid.
    """
    
    def validate_python_code(self, code: str) -> bool:
        """
        Validates the provided Python code for syntax or structural errors.
        
        Args:
            code (str): The Python code to validate.

        Returns:
            bool: True if the code is valid, False otherwise.
        """
        try:
            # Parse the code into an AST (Abstract Syntax Tree) to check for syntax errors.
            ast.parse(code)
            logging.info("Python code is syntactically valid.")
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error in code: {e}")
            logging.error(traceback.format_exc())
            return False

    def compile_python_code(self, code: str) -> bool:
        """
        Attempts to compile the provided Python code to ensure it's syntactically valid.
        
        Args:
            code (str): The Python code to compile.

        Returns:
            bool: True if the code compiles without error, False otherwise.
        """
        try:
            compiled_code = compile(code, "<string>", "exec")
            logging.info(f"Python code compiled successfully: {compiled_code}")
            return True
        except Exception as e:
            logging.error(f"Compilation error: {e}")
            logging.error(traceback.format_exc())
            return False