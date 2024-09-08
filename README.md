# GameWeaverAI

GameWeaverAI is an intelligent game generation system that allows users to generate and play simple games such as Tic-Tac-Toe. The application uses Retrieval-Augmented Generation (RAG) and a custom RAG pipeline to dynamically create games from game rule documents stored in a vector database. The system enables users to upload PDF files containing game rules and then generate code for the game based on those rules. Players can choose between single-player (AI opponent) and multiplayer modes.

## Features
- **Game Rule Ingestion**: Upload PDF documents that contain game rules, and the system will ingest the document into a vector database.
- **Dynamic Game Generation**: Generates Python code for the game based on the ingested rules and allows users to play the game directly.
- **RAG Pipeline**: Uses Retrieval-Augmented Generation to retrieve and generate game rules on demand.
- **AI Opponent**: In single-player mode, the system generates an AI opponent to play against the user.
- **Streamlit UI**: A user-friendly web interface built using Streamlit to interact with the system.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Environment Variables](#environment-variables)
5. [How it Works](#how-it-works)
6. [Acknowledgements](#acknowledgements)

## Project Structure
```
├── src
│   ├── controllers
│   │   ├── executor.py            # Executes the game logic and pipeline
│   │   ├── prompts.py             # Handles prompt generation for code and game rules
│   ├── models
│   │   ├── hf_models_manager.py   # Manages Hugging Face model loading
│   │   ├── llm_engine.py          # Handles LLM (JarvisLabs/OpenAI) service integration
│   ├── rag
│   │   ├── ingest_data.py         # Handles ingestion of PDF documents into ChromaDB
│   │   ├── retrieve_data.py       # Handles retrieval of metadata from ChromaDB
│   │   ├── rag_pipeline_base.py   # Base class for the RAG pipeline (embedding, ingestion, retrieval)
│   ├── UI
│   │   ├── streamlit_app.py       # Main Streamlit app file, handles routing
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

## Usage

### 1. **Upload Game Rules (Admin Panel)**
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

### 3. **Generate Game (To be implemented)**
   - In the future, users will be able to select a game and generate a playable game instance dynamically from the ingested rules.

## Environment Variables

The project relies on a `.env` file for key configuration settings. Ensure your `.env` file is set up correctly with the following variables:

```bash
# Root directory of the project
ROOT_PATH=/path/to/project/root

# Directory paths
DOCS_PATH=data/uploads
VECTORSTORE_PATH=vectorstore
MODELS_BASE_DIR=models

# Model and embeddings
EMBEDDING_MODEL=WhereIsAI/UAE-Large-V1

# ChromaDB Collection
CHROMADB_COLLECTION=game_rules
```

## How it Works

1. **RAG Pipeline**:
   - The system ingests game rule PDFs using the RAG pipeline and splits the document into sections. 
   - Each section is chunked, embedded using Hugging Face models, and stored in a vector database (ChromaDB).

2. **Game Generation**:
   - Game rules are retrieved from ChromaDB based on user input (e.g., Tic-Tac-Toe).
   - In future implementations, the system will generate Python code dynamically from these rules to create a playable game.

3. **AI Opponent**:
   - In single-player mode, an AI agent will be generated to play against the user.

## Acknowledgements

- **[Hugging Face](https://huggingface.co/)**: For providing the model for embedding the game rule documents.
- **[Streamlit](https://streamlit.io/)**: For the web interface to interact with the RAG pipeline.
- **[ChromaDB](https://www.trychroma.com/)**: For vector storage and retrieval.
- **LangChain**: For text chunking and recursive character splitting.