# Exercises

Short experiments, each with something to notice. They assume the playground is
running (`uv run python -m playground.web.app`) and `llama3.2` is installed.

Where a number is quoted below, it was measured on an Apple Silicon laptop with
`llama3.2` (3.2B, Q4_K_M). Yours will differ — the *shape* of the result is the
point, not the digits.

---

## 1. What temperature actually does

Set **Seed** to `42` and **Temperature** to `0`. Send "Write one sentence about
the sea." three times, clearing the conversation between each.

Now set Temperature to `1.5` and repeat.

> **Notice:** at 0 with a fixed seed the reply is identical every time — the
> model is not "creative", it is sampling, and you have pinned the sample. At
> 1.5 it varies and starts to wander. Temperature is not a personality dial; it
> widens the distribution the next token is drawn from.

Then try Temperature `0` with the seed *cleared*. Still identical? Mostly —
which tells you how little randomness is left at 0.

---

## 2. Watch the context window fill

Turn the statistics strip on and watch **context** after each reply. Have a
six-or-seven turn conversation.

> **Notice:** the context figure climbs with every turn, because the whole
> history is re-sent each time. You are paying for the entire conversation on
> every single turn — that is why long chats get slower and more expensive.

Now set **Context window (num_ctx)** to `512` and keep going.

> **Notice:** once the conversation exceeds the window, the earliest turns fall
> out and the model stops being able to recall them. Ask it something you said
> at the start to confirm. Nothing errors; the memory simply ends.

---

## 3. Free text versus a schema

Ask, in plain chat: *"Classify this ticket into category, priority and
sentiment: 'I was charged twice this month.'"*

Now tick **Force JSON schema**, paste the contents of
`examples/schemas/ticket.json`, and ask again with just the ticket text.

> **Notice:** the free-text answer is readable but differently shaped every
> time — prose, a list, extra commentary. The schema-constrained answer is
> always parseable and always uses the allowed values. The `enum` constraints
> are enforced during generation, so the model *cannot* invent a fourth
> priority.

Then measure whether it is also more *accurate*:

```bash
uv run slm eval --input examples/support_tickets.jsonl \
                --schema examples/schemas/ticket.json \
                --models llama3.2 --temperatures 0
```

---

## 4. Does temperature help extraction?

```bash
uv run slm eval --input examples/support_tickets.jsonl \
                --schema examples/schemas/ticket.json \
                --models llama3.2 --temperatures 0,0.7,1.5
```

> **Notice:** accuracy falls as temperature rises — measured here: 73% at 0,
> 68% at 1.0. For a task with one right answer, sampling more widely can only
> hurt. Creativity settings are task-dependent, and this task does not want any.

Compare every row against the **baseline** line in the table. That is the score
from always guessing the commonest label — 37% on this dataset. A model that
cannot beat it has not learned anything.

---

## 5. What reasoning costs

Requires `ollama pull deepseek-r1`.

In the playground, tick **Compare two models**, put `llama3.2` on the left and
`deepseek-r1` on the right, and ask: *"What is 17 times 3? Answer with just the
number."*

> **Notice:** both answer `51`. But expand the **Reasoning** block on the right
> and look at the statistics: measured here, llama3.2 spent **2 output tokens
> in 6.4s**, deepseek-r1 spent **540 output tokens in 27.7s** — for the same
> answer. Thinking tokens are counted in the same `eval_count`, which is why
> tokens/sec is not comparable between a reasoning and a non-reasoning model.

Then check whether the thinking buys accuracy on a real task:

```bash
uv run slm eval --input examples/support_tickets.jsonl \
                --schema examples/schemas/ticket.json \
                --models llama3.2,deepseek-r1 --temperatures 0 --limit 8
```

> **Notice:** measured here, the 3.2B model scored **88%** and the 8.2B
> reasoning model **79%**, at 0.61s versus 27.65s per row. Bigger is not
> automatically better, and "it reasons" is not automatically an advantage.
> This is the whole argument for measuring instead of guessing.

---

## 6. Hiding a control is not security

Start the app with an ability switched off:

```bash
FEATURE_COMPARE=0 uv run python -m playground.web.app
```

The compare checkbox is gone from the interface. Now ask for it anyway:

```bash
curl -i -X POST localhost:5000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"compare":true}'
```

> **Notice:** `409`, with an error explaining which flag to set. The interface
> hid the control, but that is only presentation — anyone can open a terminal.
> The rule that actually holds is the one the server enforces. Check
> `GET /api/config` to see what the interface was told.

Now try `FEATURE_STATS=0` instead and send a normal message.

> **Notice:** this one does *not* fail. Statistics are displayed, not
> requested, so the flag strips them from the response rather than refusing it.
> Not every "off" means the same thing.

---

## 7. Cold starts dominate

Stop Ollama and start it again, so nothing is loaded. Send a short message and
watch **load** in the statistics strip. Send the same message again.

> **Notice:** the first call spends seconds loading the model and milliseconds
> generating; the second spends almost nothing loading. This is why
> `slm eval` warms each model before timing it — without that, the first row of
> every sweep would be measuring disk speed rather than the model.

---

## Going further

- Write your own schema and dataset: any JSONL with a `text` field works, and
  add `labels` to make it scorable.
- Add a field to `examples/schemas/ticket.json` and see which fields the model
  finds hard. Per-field accuracy is in the `eval` table.
- Try `--repeat 3` on a sweep to see how much run-to-run variance there is at
  a given temperature.
