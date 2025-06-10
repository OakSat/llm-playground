from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os

app = Flask(__name__, static_url_path='', static_folder='../frontend')
CORS(app)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")


@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    try:
        result = subprocess.run(
            ["ollama", "run", OLLAMA_MODEL, user_input],
            capture_output=True,
            text=True,
            timeout=30
        )
        return jsonify({"response": result.stdout})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
# TODO: switch to uv, stop using pip