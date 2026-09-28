// Wiring: boot, read the controls, run a turn.

import { ApiError, chat, getConfig, getHealth, getModels, streamChat } from "./api.js";
import {
  DEFAULT_SCHEMA,
  addTurn,
  buildMessages,
  canThink,
  clearHistory,
  createState,
  findModel,
  sameModel,
} from "./state.js";
import {
  addAssistantMessage,
  addUserMessage,
  clearLane,
  describeModel,
  el,
  fillModelSelect,
  setLaneTitle,
  show,
  showBanner,
} from "./ui.js";

const state = createState();
let inFlight = null;

boot();

async function boot() {
  el("schema").value = DEFAULT_SCHEMA;
  wireControls();

  try {
    const config = await getConfig();
    state.features = config.features;
    state.defaults = config.defaults;
    applyFeatures(config.features);
  } catch (error) {
    showBanner(`Could not load configuration: ${error.message}`);
    return;
  }

  await refreshHealth();
  await refreshModels();
  updateModelDependentControls();
}

// Hide the control for every ability that is switched off. The server
// refuses those abilities regardless; this only avoids offering them.
function applyFeatures(features) {
  for (const node of document.querySelectorAll("[data-feature]")) {
    if (features[node.dataset.feature] === false) node.hidden = true;
  }
}

function featureOn(name) {
  return state.features[name] !== false;
}

async function refreshHealth() {
  try {
    const health = await getHealth();
    showBanner(health.ok ? "" : `Ollama is unreachable at ${health.ollama.host}.`);
  } catch (error) {
    showBanner(
      error.status === 503
        ? "Ollama is unreachable. Start it with `ollama serve`."
        : `Health check failed: ${error.message}`,
    );
  }
}

async function refreshModels() {
  try {
    const { models } = await getModels();
    state.models = models;
    fillModelSelect(el("model"), models, state.defaults.model, sameModel);
    fillModelSelect(el("model-b"), models, secondLaneDefault(models), sameModel);
  } catch (error) {
    showBanner(`Could not list models: ${error.message}`);
  }
}

// The second lane should differ from the first, but defaulting to
// whichever model happens to be listed first can land on something huge.
// Prefer the smallest alternative instead.
function secondLaneDefault(models) {
  const alternatives = models.filter((m) => !sameModel(m.name, state.defaults.model));
  const pool = alternatives.length ? alternatives : models;
  const smallest = [...pool].sort((a, b) => (a.size_bytes || 0) - (b.size_bytes || 0))[0];
  return smallest?.name;
}

function wireControls() {
  for (const id of ["temperature", "top_p"]) {
    const slider = el(id);
    const output = el(`${id}-value`);
    slider.addEventListener("input", () => {
      output.textContent = slider.value;
    });
  }

  el("model").addEventListener("change", updateModelDependentControls);
  el("model-b").addEventListener("change", updateModelDependentControls);

  el("schema-enabled").addEventListener("change", (event) => {
    show(el("schema"), event.target.checked);
  });

  el("compare").addEventListener("change", (event) => {
    const on = event.target.checked;
    show(el("model-b-control"), on);
    show(document.querySelector('.lane[data-lane="b"]'), on);
    updateModelDependentControls();
  });

  el("reset").addEventListener("click", () => {
    clearHistory(state);
    clearLane("a");
    clearLane("b");
  });

  el("composer").addEventListener("submit", (event) => {
    event.preventDefault();
    send();
  });

  el("stop").addEventListener("click", () => inFlight?.abort());

  el("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });
}

// The reasoning toggle is offered only by models that declare the
// capability. Asking anything else to think is answered with a 400. When
// comparing, either lane having the capability is enough to offer it.
function updateModelDependentControls() {
  const name = el("model").value;
  el("model-detail").textContent = describeModel(findModel(state, name));

  const supported = canThink(state, name) || (comparing() && canThink(state, el("model-b").value));
  show(el("think-control"), supported && featureOn("reasoning"));
  if (!supported) el("think").checked = false;

  updateLaneTitles();
}

function comparing() {
  return featureOn("compare") && el("compare").checked;
}

function updateLaneTitles() {
  if (!comparing()) {
    setLaneTitle("a", "");
    return;
  }
  setLaneTitle("a", el("model").value);
  setLaneTitle("b", el("model-b").value);
}

function readOptions() {
  const options = {
    temperature: Number(el("temperature").value),
    top_p: Number(el("top_p").value),
  };
  const seed = el("seed").value.trim();
  if (seed !== "") options.seed = Number(seed);
  const numCtx = el("num_ctx").value.trim();
  if (numCtx !== "") options.num_ctx = Number(numCtx);
  return options;
}

function readSchema() {
  if (!featureOn("structured_output") || !el("schema-enabled").checked) return null;
  try {
    return JSON.parse(el("schema").value);
  } catch (error) {
    throw new Error(`The JSON schema is not valid JSON: ${error.message}`);
  }
}

function setBusy(busy) {
  el("send").disabled = busy;
  show(el("stop"), busy);
}

async function send() {
  const input = el("input");
  const prompt = input.value.trim();
  if (!prompt || inFlight) return;

  let schema;
  try {
    schema = readSchema();
  } catch (error) {
    showBanner(error.message);
    return;
  }
  showBanner("");

  const lanes = comparing()
    ? [
        { lane: "a", model: el("model").value },
        { lane: "b", model: el("model-b").value },
      ]
    : [{ lane: "a", model: el("model").value }];

  input.value = "";
  inFlight = new AbortController();
  setBusy(true);

  const options = readOptions();
  const systemPrompt = el("system").value;
  const wantsThinking = featureOn("reasoning") && el("think").checked;

  for (const { lane } of lanes) {
    addTurn(state, lane, { role: "user", content: prompt });
    addUserMessage(lane, prompt);
  }

  // Each lane is an independent request, so the two run concurrently and
  // neither waits on the other.
  await Promise.all(
    lanes.map(({ lane, model }) =>
      runLane({
        lane,
        model,
        options,
        schema,
        // Only ask a model to think if it actually can; the other lane
        // may be a model without the capability.
        think: wantsThinking && canThink(state, model),
        systemPrompt,
        comparing: lanes.length > 1,
      }),
    ),
  );

  inFlight = null;
  setBusy(false);
  input.focus();
}

async function runLane({ lane, model, options, schema, think, systemPrompt, comparing }) {
  const body = {
    model,
    messages: buildMessages(state, lane, systemPrompt),
    options,
    think,
  };
  if (schema) body.format = schema;
  if (comparing) body.compare = true;

  const view = addAssistantMessage(lane, contextLimitFor(model, options));

  try {
    const content = featureOn("streaming")
      ? await runStreaming(body, view)
      : await runBlocking(body, view);
    addTurn(state, lane, { role: "assistant", content });
  } catch (error) {
    if (error.name === "AbortError") {
      view.fail("Stopped.");
      return;
    }
    view.fail(describeError(error));
  }
}

async function runStreaming(body, view) {
  let content = "";
  for await (const { event, data } of streamChat(body, inFlight.signal)) {
    if (event === "error") throw new ApiError(data.error, data.status);
    if (data.thinking) view.appendThinking(data.thinking);
    if (data.content) {
      content += data.content;
      view.appendContent(data.content);
    }
    if (event === "done") view.finish(data.stats);
  }
  return content;
}

async function runBlocking(body, view) {
  const data = await chat(body, inFlight.signal);
  if (data.thinking) view.appendThinking(data.thinking);
  view.appendContent(data.content);
  view.finish(data.stats);
  return data.content;
}

// The effective window: an explicit num_ctx overrides the model's own.
function contextLimitFor(model, options) {
  return options.num_ctx || findModel(state, model)?.context_length || null;
}

function describeError(error) {
  if (error.status === 409) return `${error.message}`;
  if (error.status === 503) return "Ollama is unreachable. Is `ollama serve` running?";
  return error.message || String(error);
}
