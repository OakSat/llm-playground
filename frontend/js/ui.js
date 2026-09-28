// Rendering. Everything user-supplied goes in through textContent, never
// innerHTML, so a model that emits markup cannot inject it into the page.

export const el = (id) => document.getElementById(id);

export function show(node, visible) {
  if (node) node.hidden = !visible;
}

export function showBanner(message) {
  const banner = el("banner");
  banner.textContent = message;
  show(banner, Boolean(message));
}

export function clearLane(lane) {
  el(`messages-${lane}`).replaceChildren();
}

function scrollToEnd(container) {
  container.scrollTop = container.scrollHeight;
}

export function addUserMessage(lane, text) {
  const container = el(`messages-${lane}`);
  const wrapper = document.createElement("div");
  wrapper.className = "msg user";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  wrapper.append(bubble);
  container.append(wrapper);
  scrollToEnd(container);
}

// Returns a handle the caller drives as tokens arrive.
export function addAssistantMessage(lane, contextLimit = null) {
  const container = el(`messages-${lane}`);

  const wrapper = document.createElement("div");
  wrapper.className = "msg assistant";

  // Shown until the first token. A cold model can take seconds to load,
  // so silence here would look like a hang.
  const waiting = document.createElement("div");
  waiting.className = "waiting";
  const spinner = document.createElement("div");
  spinner.className = "spinner";
  waiting.append(spinner, document.createTextNode("waiting for the first token…"));

  const thinking = document.createElement("details");
  thinking.className = "thinking";
  thinking.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";
  const scratchpad = document.createElement("div");
  scratchpad.className = "scratchpad";
  thinking.append(summary, scratchpad);

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.hidden = true;

  const stats = document.createElement("div");
  stats.className = "stats";
  stats.hidden = true;

  wrapper.append(waiting, thinking, bubble, stats);
  container.append(wrapper);
  scrollToEnd(container);

  let answer = "";
  let reasoning = "";

  return {
    appendContent(text) {
      if (!text) return;
      answer += text;
      waiting.hidden = true;
      bubble.hidden = false;
      bubble.textContent = answer;
      scrollToEnd(container);
    },
    appendThinking(text) {
      if (!text) return;
      reasoning += text;
      waiting.hidden = true;
      thinking.hidden = false;
      scratchpad.textContent = reasoning;
      scrollToEnd(container);
    },
    finish(statsData) {
      waiting.hidden = true;
      if (!answer) {
        bubble.hidden = false;
        bubble.textContent = "(empty reply)";
      }
      if (statsData) {
        renderStats(stats, statsData, contextLimit);
        stats.hidden = false;
      }
      scrollToEnd(container);
      return answer;
    },
    fail(message) {
      waiting.hidden = true;
      wrapper.classList.add("error");
      bubble.hidden = false;
      bubble.textContent = answer ? `${answer}\n\n${message}` : message;
      scrollToEnd(container);
    },
  };
}

function renderStats(node, stats, contextLimit) {
  const entries = [];
  const speed = stats.tokens_per_second;
  if (speed != null) entries.push(["speed", `${speed.toFixed(1)} tok/s`]);
  if (stats.completion_tokens != null) entries.push(["out", `${stats.completion_tokens} tok`]);
  if (stats.prompt_tokens != null) entries.push(["in", `${stats.prompt_tokens} tok`]);
  if (stats.context_used != null) {
    entries.push(["context", formatContext(stats.context_used, contextLimit)]);
  }
  if (stats.load_seconds) entries.push(["load", `${stats.load_seconds.toFixed(2)}s`]);
  if (stats.total_seconds != null) entries.push(["total", `${stats.total_seconds.toFixed(2)}s`]);

  node.replaceChildren();
  for (const [label, value] of entries) {
    const item = document.createElement("span");
    const strong = document.createElement("b");
    strong.textContent = value;
    item.append(strong, document.createTextNode(` ${label}`));
    node.append(item);
  }
}

// Shown against the window in force, so filling it up is visible.
function formatContext(used, limit) {
  return limit ? `${used}/${limit.toLocaleString()} tok` : `${used} tok`;
}

export function fillModelSelect(select, models, selected, matches) {
  select.replaceChildren();
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.name;
    const size = model.parameter_size ? ` · ${model.parameter_size}` : "";
    option.textContent = `${model.name}${size}`;
    select.append(option);
  }
  const chosen = models.find((m) => matches(m.name, selected));
  if (chosen) select.value = chosen.name;
}

export function describeModel(model) {
  if (!model) return "";
  const parts = [];
  if (model.parameter_size) parts.push(model.parameter_size);
  if (model.quantization) parts.push(model.quantization);
  if (model.context_length) parts.push(`${model.context_length.toLocaleString()} ctx`);
  if (model.capabilities.length) parts.push(model.capabilities.join(", "));
  return parts.join(" · ");
}

export function setLaneTitle(lane, text) {
  const title = document.querySelector(`.lane[data-lane="${lane}"] .lane-title`);
  title.textContent = text;
  show(title, Boolean(text));
}
