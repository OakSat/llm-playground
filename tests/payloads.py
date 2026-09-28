"""Recorded Ollama payloads used by the test suite.

The Ollama HTTP layer is mocked, so the whole suite runs with nothing
installed and no model loaded -- which is what lets CI stay green on a
plain runner. The payloads below are trimmed copies of real responses
captured from Ollama 0.34.1, so the parsing tests exercise the actual
wire format rather than an idealised one.
"""

import json

HOST = "http://ollama.test"

# Captured from `POST /api/chat` with stream=false.
CHAT_RESPONSE = {
    "model": "llama3.2",
    "created_at": "2026-09-28T08:59:02.268036Z",
    "message": {"role": "assistant", "content": "Hello there!"},
    "done": True,
    "done_reason": "stop",
    "total_duration": 2705802709,
    "load_duration": 2398213667,
    "prompt_eval_count": 31,
    "prompt_eval_duration": 233471000,
    "eval_count": 4,
    "eval_duration": 55323000,
}

# Captured from a reasoning model: the scratchpad arrives separately
# from the answer.
THINKING_RESPONSE = {
    "model": "deepseek-r1",
    "message": {
        "role": "assistant",
        "content": "\n4",
        "thinking": "\nFirst, the user asked: 2+2. That is 4.\n",
    },
    "done": True,
    "done_reason": "stop",
    "eval_count": 186,
    "eval_duration": 1000000000,
}

TAGS_RESPONSE = {
    "models": [
        {
            "name": "llama3.2:latest",
            "size": 2019393189,
            "details": {
                "parameter_size": "3.2B",
                "quantization_level": "Q4_K_M",
                "context_length": 131072,
            },
        },
        {
            "name": "deepseek-r1:latest",
            "size": 5225376256,
            "details": {
                "parameter_size": "8.2B",
                "quantization_level": "Q4_K_M",
                "context_length": 131072,
            },
        },
    ]
}

CAPABILITIES = {
    "llama3.2:latest": ["completion", "tools"],
    "deepseek-r1:latest": ["tools", "thinking", "completion"],
}


def ndjson(*payloads: dict[str, object]) -> str:
    """Render payloads the way Ollama streams them: one JSON object per line."""
    return "".join(f"{json.dumps(payload)}\n" for payload in payloads)


STREAM_BODY = ndjson(
    {"message": {"role": "assistant", "content": "1"}, "done": False},
    {"message": {"role": "assistant", "content": ", "}, "done": False},
    {"message": {"role": "assistant", "content": "2"}, "done": False},
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "total_duration": 2243342000,
        "load_duration": 1869432750,
        "prompt_eval_count": 31,
        "eval_count": 14,
        "eval_duration": 258452000,
    },
)
