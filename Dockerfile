# Use the official Python slim image to reduce size
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only requirements.txt first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies without cache to save space
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Clean up any apt caches and temporary files
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set environment variables to ensure they can be passed during runtime
ENV ROOT_PATH=/app \
    DOCS_PATH=/app/data/uploads \
    VECTORSTORE_PATH=/app/vectorstore \
    MODELS_BASE_DIR=/app/models

# Expose the Streamlit port (8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/UI/streamlit_app.py"]