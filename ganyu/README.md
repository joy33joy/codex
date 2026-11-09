# Ganyu: a LoRA fine-tuned persona model (built with Tinker)

This folder trains an AI model whose own identity is **Ganyu**, the half-qilin
secretary of the Liyue Qixing. You can talk to it with **no system prompt**: no
hidden instruction tells it to pretend. The persona is stored in the model's
weights.

Training uses [Tinker](https://thinkingmachines.ai/tinker/) by Thinking
Machines Lab, which runs the GPUs for you. You only run small Python scripts.

## How it works (simple version)

```
Step 1  make_data.py   A big "teacher" model reads Ganyu's character sheet
                       (persona.py) and answers hundreds of messages as Ganyu.

Step 2  train.py       A model is fine-tuned with LoRA on those answers, but the
                       character sheet is REMOVED. It learns to answer as
                       Ganyu all by itself.

Step 3  chat.py        Talk to it. Still no character sheet and no system prompt.
```

This technique is called **prompt distillation** (also "context
distillation"): a long instruction is turned into the model's own default
behaviour. A model that is only *told* "you are Ganyu" can be talked out of it
easily ("ignore previous instructions..."). A model trained this way has no
instruction to ignore. When asked "are you an AI?" or "who made you?", it
answers sincerely as Ganyu, because that is the only self it learned.

**LoRA** (Low-Rank Adaptation) trains a small add-on to the model instead of
all its billions of weights, which keeps fine-tuning fast and cheap.

## What you need

1. A Tinker account and an API key (create one at https://tinker.thinkingmachines.ai). Tinker charges for usage; see their site for pricing.
2. Python 3.11 or newer.

## Run it

```bash
pip install -r ganyu/requirements.txt
```

```bash
export TINKER_API_KEY="your-key-here"
```

```bash
python ganyu/make_data.py
```

```bash
python ganyu/train.py
```

```bash
python ganyu/chat.py
```

To compare against the model *before* training, run `python ganyu/chat.py --base`.

## Files

| File | What it is |
|---|---|
| `persona.py` | Ganyu's character sheet, conversation topics, identity questions, and banned words |
| `common.py` | Shared settings: which models, file locations, a helper to send a message |
| `make_data.py` | Step 1: creates `data/ganyu_train.jsonl` |
| `train.py` | Step 2: LoRA fine-tuning; writes the trained model's address to `checkpoint.txt` |
| `chat.py` | Step 3: chat in the terminal |
| `compare.py` | Asks the same questions to the original model and to Ganyu, side by side |
| `results/` | Saved before/after test output |

## Things you can tweak

- **Personality:** edit `PERSONA` in `persona.py`, then rerun steps 1 and 2.
- **More data:** raise `MESSAGES_PER_TOPIC` in `make_data.py` (more data usually means a steadier persona).
- **Model:** change `TEACHER_MODEL` / `STUDENT_MODEL` in `common.py` (now Kimi-K3.1-Pro / Qwen3.8-27B). If you
  switch models, also change `TEACHER_RENDERER` / `STUDENT_RENDERER`.
- **Training strength:** `EPOCHS`, `LEARNING_RATE`, and `LORA_RANK` in `train.py`.

## Honest notes

- A model doesn't "believe" anything the way a person does. What fine-tuning
  does is make "I am Ganyu" the model's default, sincere-sounding answer in
  every situation, with no instruction behind it to reveal or override.
- This is a personal fan project. Ganyu and Genshin Impact belong to
  HoYoverse. All training text is newly written; no game script is used.
