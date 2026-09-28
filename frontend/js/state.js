// Conversation and control state.
//
// History lives here, in the browser, and is sent in full with every
// request: the server keeps nothing between turns. That is what makes the
// growth of the context window visible, and what lets two comparison
// lanes diverge without interfering with each other.

export const DEFAULT_SCHEMA = JSON.stringify(
  {
    type: "object",
    properties: {
      name: { type: "string" },
      age: { type: "integer" },
      city: { type: "string" },
    },
    required: ["name", "age", "city"],
  },
  null,
  2,
);

export function createState() {
  return {
    features: {},
    defaults: {},
    models: [],
    // One message list per lane, so a comparison can diverge.
    history: { a: [], b: [] },
  };
}

// Ollama reports tagged names ("llama3.2:latest") while configuration
// usually spells the model untagged ("llama3.2"), so matching tolerates
// a missing tag on either side.
export function sameModel(a, b) {
  if (!a || !b) return false;
  return a === b || a.split(":")[0] === b.split(":")[0];
}

export function findModel(state, name) {
  return (
    state.models.find((model) => model.name === name) ||
    state.models.find((model) => sameModel(model.name, name)) ||
    null
  );
}

export function canThink(state, name) {
  const model = findModel(state, name);
  return Boolean(model && model.capabilities.includes("thinking"));
}

export function addTurn(state, lane, message) {
  state.history[lane].push(message);
}

export function clearHistory(state) {
  state.history.a = [];
  state.history.b = [];
}

// The system prompt is not stored in history: it is re-applied each turn
// so editing it takes effect immediately rather than only for new chats.
export function buildMessages(state, lane, systemPrompt) {
  const messages = [];
  if (systemPrompt.trim()) {
    messages.push({ role: "system", content: systemPrompt.trim() });
  }
  for (const turn of state.history[lane]) {
    messages.push({ role: turn.role, content: turn.content });
  }
  return messages;
}
