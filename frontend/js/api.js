// Talking to the playground's own API.
//
// Streaming uses fetch() with a stream reader rather than EventSource,
// because EventSource can only issue GET requests and a chat turn has to
// POST its messages. The frames on the wire are still ordinary SSE.

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

const JSON_HEADERS = { "Content-Type": "application/json" };

async function request(url, options = {}) {
  let response;
  try {
    response = await fetch(url, options);
  } catch (cause) {
    throw new ApiError("Could not reach the playground server.", 0);
  }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new ApiError(data.error || response.statusText, response.status);
  }
  return data;
}

export const getConfig = () => request("/api/config");
export const getModels = () => request("/api/models");
export const getHealth = () => request("/api/health");

export function chat(body, signal) {
  return request("/api/chat", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });
}

// Yields { event, data } for each SSE frame the server sends.
export async function* streamChat(body, signal) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok) {
    // A refusal (a disabled ability, a bad request) arrives before the
    // stream starts, so it still carries a real status code.
    const data = await response.json().catch(() => ({}));
    throw new ApiError(data.error || response.statusText, response.status);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    // Frames are separated by a blank line and may straddle chunks.
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const parsed = parseFrame(frame);
      if (parsed) yield parsed;
    }
  }
}

function parseFrame(frame) {
  let event = "message";
  let data = "";
  for (const line of frame.split("\n")) {
    if (line.startsWith("event: ")) event = line.slice(7);
    else if (line.startsWith("data: ")) data += line.slice(6);
  }
  if (!data) return null;
  try {
    return { event, data: JSON.parse(data) };
  } catch {
    return null;
  }
}
