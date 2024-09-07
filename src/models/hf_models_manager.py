"""
####################################################################################
#####                File: src/models/hf_models_manager.py                     #####
#####                         Author: Ravi Dhir                                #####
#####                      Created on: 09/05/2024                              #####
#####         Hugging Face Models Manager to Load Models from HF               #####
####################################################################################
"""

import os

import yaml
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Load the config.yaml file
with open(os.path.join(os.path.dirname(__file__), '..', '..', 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
load_dotenv()
 
class HFModelsManager:
    def __init__(self, repo_name: str, model_path: str = None, model_type: str = 'embedding'):
        """
        Initializes the Huggingface Model Manager by connecting to Huggingface and preparing the model.

        Args:
            repo_name (str): The Huggingface repository name (e.g., "bert-base-uncased").
            model_path (str, optional): The local base path where the model will be downloaded. 
                                        Defaults to MODEL_BASE_DIR environment variable or './models'.
            model_type (str, optional): Type of model ('embedding' or 'causal_lm'). Defaults to 'embedding'.
        """
        self.repo_name = repo_name
        self.model_type = model_type
        self.model_base_dir = model_path or os.getenv('MODEL_BASE_DIR', './models')
        self.token = os.getenv('HUGGINGFACE_TOKEN')

        if not self.token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")
        
        # Connect to Huggingface using the token via login
        login(token=self.token)
        print("Connected to Huggingface")

        # Ensure the model directory is prepared and the model is available locally
        self.model_dir = self.ensure_model_available(self.repo_name, self.model_base_dir)

    def ensure_model_available(self, repo_name: str, local_dir: str) -> str:
        """
        Ensure that the model is available locally under model_path/repo_name.
        If not, download it from the Huggingface repo.

        Args:
            repo_name (str): The Huggingface model repository name.
            local_dir (str): The local base directory where the repo directory will be created.

        Returns:
            str: The path to the local model directory (model_path/repo_name).
        """
        try:
            # Define the specific path for the model (model_path/repo_name)
            repo_local_dir = os.path.join(local_dir, repo_name)

            # Create the repo_name directory if it doesn't exist
            if not os.path.exists(repo_local_dir):
                os.makedirs(repo_local_dir)

            # Download the model to the repo_local_dir
            model_path = snapshot_download(repo_id=repo_name, local_dir=repo_local_dir)
            print(f"Model '{repo_name}' is available at {model_path}")
            return model_path
        except Exception as e:
            print(f"Failed to download or locate model: {e}")
            raise

    def initialize_model(self):
        """
        Initialize the model and tokenizer from the local model directory.

        Returns:
            Tuple[AutoModel, AutoTokenizer]: The initialized model and tokenizer.
        """
        try:
            if self.model_type == 'embedding':
                model = AutoModel.from_pretrained(self.model_dir)
            elif self.model_type == 'causal_lm':
                model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            else:
                raise ValueError(f"Invalid model type: {self.model_type}. Use 'embedding' or 'causal_lm'.")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            print(f"Model and tokenizer for '{self.repo_name}' initialized successfully.")
            return model, tokenizer
        except Exception as e:
            print(f"Failed to initialize model: {e}")
            raise


if __name__ == "__main__":
    repo_name = config['models']['EMBEDDING_MODEL']
    # Instantiate the HFModelsManager
    manager = HFModelsManager(repo_name=repo_name, model_type='embedding')

    # Initialize the model and tokenizer
    model, tokenizer = manager.initialize_model()
    print(model, tokenizer)
