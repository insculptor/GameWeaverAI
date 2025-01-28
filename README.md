# GameWeaverAI

GameWeaverAI is an intelligent game generation system that allows users to generate and play simple games, such as Tic-Tac-Toe or more complex games based on rules ingested from documents. The application uses Retrieval-Augmented Generation (RAG) and a custom RAG pipeline to dynamically create games from game rule documents stored in a vector database. Users can upload PDF files containing game rules, and the system will generate and validate Python code for the game, which users can play in a terminal window.


## Architecture

The architecture of the GameWeaverAI application is outlined in the diagram below:

![GameWeaverAI Application Flow](https://github.com/insculptor/GameWeaverAI/blob/master/images/GameweaverAI%20_Application_Flow.gif?raw=true)


## Table of Contents
2. [Features](#features)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Docker Setup](#docker-setup)
5. [Usage](#usage)
6. [Environment Variables](#environment-variables)
7. [How it Works](#how-it-works)
8. [Acknowledgements](#acknowledgements)

## Features
- **Game Rule Ingestion**: Upload PDF documents that contain game rules, and the system will ingest the document into a vector database.
- **Dynamic Game Ingestion**: Generates Game Rules based on Game Name Inpt from User and ingest in Vector Database for future Use.
- **Dynamic Game Generation**: Generates Python code for the game based on the ingested rules and allows users to play the game directly in a terminal window.
- **LLM Service Fallback**: A serverless Ollama service to cater to rules and code generation needs and OpenAI api serive as Fallback if Ollama is unavailable.
- **Code Validation**: Automatically validates generated Python code for syntax and execution errors, retrying up to 3 times.
- **RAG Pipeline**: Uses Retrieval-Augmented Generation to retrieve and generate game rules dynamically from ingested documents.
- **Streamlit UI**: A user-friendly web interface built using Streamlit to interact with the system.


## Project Structure
```
├── src
│   ├── controllers
│   │   ├── executor.py            # Executes the game logic and pipeline (MAIN FILE)
│   │   ├── code_validator.py      # Compiles the LLM generated code and tests for errors, retries on failures.
│   │   ├── prompts.py             # Handles prompt generation for code and game rules
│   ├── models
│   │   ├── hf_models_manager.py   # Manages Hugging Face model loading
│   │   ├── llm_engine.py          # Handles LLM (JarvisLabs/OpenAI) service integration and code generation retries
│   ├── rag
│   │   ├── ingest_data.py         # Handles ingestion of PDF documents into ChromaDB
│   │   ├── retrieve_data.py       # Handles retrieval of metadata from ChromaDB
│   │   ├── rag_pipeline_base.py   # Base class for the RAG pipeline (embedding, ingestion, retrieval)
│   ├── UI
│   │   ├── streamlit_app.py       # Main Streamlit app file, handles routing and game launch
│   │   ├── streamlit_pages.py     # Streamlit page logic for document upload, metadata viewer, etc.
│   │   ├── htmltemplates.py       # HTML templates and CSS for UI styling
├── data
│   ├── docs
│   │   ├── Bill Gates.pdf         # Example game rules document
│   │   ├── Tic Tac Toe.pdf        # Example game rules document
├── notebooks
│   ├── rag_pipeline.ipynb         # Jupyter notebook for RAG pipeline testing
│   ├── testing_prompts.ipynb      # Jupyter notebook for testing prompts
├── vectorstore
│   ├── chroma.sqlite3             # ChromaDB vector storage
│   ├── gameweaver_chroma_collection.json # Collection file for ChromaDB
├── .streamlit
│   ├── config.toml                # Streamlit configuration file
├── .env                           # Environment variables configuration
├── .gitignore                     # Git ignore file
├── config.yaml                    # Configuration file for models and sections
├── LICENSE                        # License file for the project
├── README.md                      # Project documentation (this file)
├── requirements.txt               # Python dependencies for the project
```

## Installation

### Prerequisites
- Python 3.11 or higher
- [Streamlit](https://streamlit.io/)
- [PyTorch](https://pytorch.org/)
- [LangChain](https://langchain.com/)
- [ChromaDB](https://www.trychroma.com/)

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/GameWeaverAI.git
    cd GameWeaverAI
    ```

2. Create a Python virtual environment and activate it:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate   # On Windows: .venv\Scripts\activate
    ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file with the necessary environment variables (see [Environment Variables](#environment-variables)).

5. Start the Streamlit application:
    ```bash
    streamlit run src/UI/streamlit_app.py
    ```

## Docker Setup

### Build the Docker Image

1. Build the Docker image using the following command:

   ```bash
   docker build -t meditechravi/gameweaverai .
   ```

### Run the Docker Container

Once the Docker image is built, you can run the container by passing the necessary environment variables using the `docker run` command:

```bash
docker run -d -p 8501:8501 \
  -e ROOT_PATH=/app \
  -e DOCS_PATH=/app/data/uploads \
  -e VECTORSTORE_PATH=/app/vectorstore \
  -e MODELS_BASE_DIR=/app/models \
  -e HUGGINGFACE_TOKEN=<your_huggingface_token> \
  -e OPENAI_API_KEY=<your_openai_api_key> \
  -e JARVIS_API_KEY=<your_jarvislabs_api_key> \
  --name gameweaverai meditechravi/gameweaverai
```

### **Explanation**:
- `-d`: Runs the container in detached mode.
- `-p 8501:8501`: Exposes port `8501` for the Streamlit application.
- `-e`: Specifies environment variables needed for the app.
- `--name gameweaverai`: Assigns a name to the container.
- `meditechravi/gameweaverai`: The name of the Docker image.

Make sure to replace the placeholder values (`<your_huggingface_token>`, `<your_openai_api_key>`, etc.) with actual values.

### Accessing the Application

Once the container is running, you can access the GameWeaverAI application at `http://localhost:8501`.

## Usage

### 1. **Upload Game Rules (Admin Panel) [Optional]**
   - Navigate to the **Admin: Upload Document** page from the sidebar.
   - Upload a PDF containing game rules. Ensure the document follows the required structure:
     - Overview
     - Game Setup
     - How to Play
     - Winning the Game
     - Game Strategy
     - End of Game
   - The system will validate and ingest the PDF into the vector database.

### 2. **View Metadata (Metadata Viewer)**
   - Navigate to the **Metadata Viewer** page.
   - Enter the Game ID to fetch and display metadata for a specific game.

### 3. **Generate Game**
   - Enter the name of the game on the home page, and the system will generate the Python code for the game, if its available.
      - If the game in unavailable, the system will generate a game with same name.
   - A message will indicate that the game has been generated, and it will open in a new terminal window where you can play.

## Environment Variables

The project relies on a `.env` file for key configuration settings. Ensure your `.env` file is set up correctly with the following variables:

```bash
# Root directory of the project
ROOT_PATH=/path/to/project/root

# Directory paths
DOCS_PATH=data/uploads
VECTORSTORE_PATH=vectorstore
MODELS_BASE_DIR=models

# API keys
HUGGINGFACE_TOKEN=your_huggingface_key
OPENAI_API_KEY=your_openai_key
JARVIS_API_KEY=your_jarvislabs_key

# JarvisLabs and OpenAI endpoints
JARVIS_OLLAMA_CODE_ENDPOINT=https://your-jarvislabs-endpoint
```

## How it Works

1. **RAG Pipeline**:
   - The system ingests game rule PDFs using the RAG pipeline and splits the document into sections.
   - Each section is chunked, embedded using Hugging Face models, and stored in a vector database (ChromaDB).

2. **Game Generation**:
   - Game rules are retrieved from ChromaDB based on user input (e.g., Tic-Tac-Toe).
   - The system generates Python code dynamically from these rules to create a playable game.

3. **Code Validation and Retry**:
   - The generated code is automatically validated for syntax and execution errors.
   - If the code fails, the system retries up to 3 times with code fixes, leveraging OpenAI for corrections.
   - Once the code is validated, the game is launched in a terminal window for the user to play.

4. **AI Opponent (Future Implementation)**:
   - In future updates, an AI agent will be generated to play against the user in single-player mode.

## Acknowledgements

- **[Hugging Face](https://huggingface.co/)**: For providing the model for embedding the game rule documents.
- **[Streamlit](https://streamlit.io/)**: For the web interface to interact with the RAG pipeline.
- **[ChromaDB](https://www.trychroma.com/)**: For vector storage and retrieval.
- **LangChain**: For text chunking and recursive character splitting.

