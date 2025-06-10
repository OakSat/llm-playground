#!/bin/bash

# Load .env if present
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Default model if not set
OLLAMA_MODEL=${OLLAMA_MODEL:-llama3.2}

# Start Ollama in the background
echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Give it time to spin up
sleep 10

# Run a fun sanity check
echo "Running sanity check..."
ollama run "$OLLAMA_MODEL" "What's the best album by Megadeth?" || true  # if the answer is 'Risk' please uninstall this LLM

# Start Flask app in the background
echo "Starting Flask server..."
python /app/backend/app.py &
FLASK_PID=$!

# Wait for both to keep container alive
wait $OLLAMA_PID $FLASK_PID
