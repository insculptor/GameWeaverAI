"""
####################################################################################
#####                    File: src/models/llm.py                                #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####     Unified LLM Service Class for JarvisLabs (primary) and OpenAI         #####
####################################################################################
"""

import os

import requests
import yaml
from dotenv import load_dotenv


class LLMService:
    """
    LLMService class that handles code and rules generation using either JarvisLabs or OpenAI.
    JarvisLabs is the primary service, and OpenAI is used as a fallback.
    """
    def __init__(self):
        """
        Initialize the LLMService class by loading the necessary environment variables and config settings.
        """
        # Load environment variables
        load_dotenv()

        # Load config.yaml for model names
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml')
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # JarvisLabs API details
        self.jarvis_api_key = os.getenv("JARVIS_API_KEY")
        self.jarvis_code_endpoint = os.getenv("JARVIS_OLLAMA_CODE_ENDPOINT")

        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = self.config['models']['OPENAI_LLM_MODEL']

        # JarvisLabs - Ollama Models
        self.jarvis_code_llm = self.config['models']['JARVIS_CODE_LLM']
        self.jarvis_rules_llm = self.config['models']['JARVIS_RULES_LLM']

    def is_jarvis_available(self):
        """
        Check if the JarvisLabs API is available by making a simple request.

        Returns:
            bool: True if JarvisLabs is available, False otherwise.
        """
        try:
            headers = {"Authorization": f"Bearer {self.jarvis_api_key}"}
            response = requests.get(self.jarvis_code_endpoint, headers=headers, timeout=5)
            return response.status_code == 200
        except requests.RequestException as e:
            print(f"JarvisLabs is unavailable: {e}")
        return False

    def generate_code_jarvis(self, prompt: str):
        """
        Generates code using JarvisLabs.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code or an error message.
        """
        headers = {
            "Authorization": f"Bearer {self.jarvis_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                self.jarvis_code_endpoint + "chat/completions",
                headers=headers,
                json={
                    "model": self.jarvis_code_llm,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Jarvis API error: {e}"

    def generate_code_openai(self, prompt: str):
        """
        Generates code using OpenAI.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code or an error message.
        """
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                "https://api.openai.com/v1/completions",
                headers=headers,
                json={
                    "model": self.openai_model,
                    "prompt": prompt,
                    "max_tokens": 256,
                    "temperature": 0.5
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["text"].strip()
        except Exception as e:
            return f"OpenAI API error: {e}"

    def generate_code(self, prompt: str):
        """
        Unified function for generating code. Tries JarvisLabs first, falls back to OpenAI if Jarvis is unavailable.

        Args:
            prompt (str): The input prompt to generate code.

        Returns:
            str: The generated code.
        """
        if self.is_jarvis_available():
            return self.generate_code_jarvis(prompt)
        else:
            return self.generate_code_openai(prompt)

    def generate_rules_jarvis(self, prompt: str):
        """
        Generates game rules using JarvisLabs.

        Args:
            prompt (str): The input prompt to generate rules.

        Returns:
            str: The generated game rules or an error message.
        """
        headers = {
            "Authorization": f"Bearer {self.jarvis_api_key}",
            "Content-Type": "application/json"
        }
        try:
            response = requests.post(
                self.jarvis_code_endpoint + "chat/completions",
                headers=headers,
                json={
                    "model": self.jarvis_rules_llm,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"Jarvis API error: {e}"

    def generate_rules(self, prompt: str):
        """
        Unified function for generating game rules. Tries JarvisLabs first, falls back to OpenAI if Jarvis is unavailable.

        Args:
            prompt (str): The input prompt to generate rules.

        Returns:
            str: The generated game rules.
        """
        if self.is_jarvis_available():
            return self.generate_rules_jarvis(prompt)
        else:
            return self.generate_code_openai(prompt)  # OpenAI fallback


# Example usage
if __name__ == "__main__":
    llm_service = LLMService()

    # Generate code example
    code_prompt = "Generate Python code for a Tic-Tac-Toe game."
    generated_code = llm_service.generate_code(code_prompt)
    print("Generated Code:\n", generated_code)

    # Generate game rules example
    rules_prompt = "Create game rules for a strategy game called 'Space Adventure'."
    generated_rules = llm_service.generate_rules(rules_prompt)
    print("Generated Game Rules:\n", generated_rules)
