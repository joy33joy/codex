# Your own Ganyu training data

Put your files **in this folder** (`ganyu/my_data/`). Use any formats you like,
and mix them freely. Example files for every format are in `examples/`.

| File name | What you write | What happens to it |
|---|---|---|
| `conversations.jsonl` | Full chats, possibly several turns long | Used exactly as written |
| `pairs.csv` | A spreadsheet with columns `user,ganyu` | Used exactly as written |
| `ganyu_lines.txt` | Only Ganyu's lines, one per line | The teacher model writes a user message that leads to each line |
| `questions.txt` | Only messages *to* Ganyu, one per line | The teacher model answers each one as Ganyu |

The easiest start is `pairs.csv`: open it in Excel, Google Sheets or Numbers
and **Save as / Download as CSV**.

## Generate 500 conversations with your own AI (automatic)

`ganyu/generate_with_my_ai.py` asks your AI for one conversation at a time,
500 times, and appends each good one to `my_data/conversations.jsonl`.
Every request gets a different random **topic**, **speaking style** for the
person, **number of exchanges** (1-3), **reply length**, and **twist**, so
the 500 conversations don't repeat each other. It throws away answers that
aren't valid JSON, that break character, or that repeat an opening line, and
asks again. You can stop it any time; running it again continues where it
left off.

Pick the command for your AI (install its package first):

```bash
pip install anthropic
export ANTHROPIC_API_KEY="your-key"
python ganyu/generate_with_my_ai.py --provider claude
```

```bash
pip install openai
export OPENAI_API_KEY="your-key"
python ganyu/generate_with_my_ai.py --provider openai --model YOUR-MODEL-NAME
```

The same `--provider openai` works for any service with an OpenAI-compatible
API. Add `--base-url`, and put that service's key in `OPENAI_API_KEY`:

| Service | Add this |
|---|---|
| Google Gemini | `--base-url https://generativelanguage.googleapis.com/v1beta/openai/` |
| DeepSeek | `--base-url https://api.deepseek.com` |
| Ollama on your computer (free, no key) | `--base-url http://localhost:11434/v1` |

Useful options: `--count 1000` (a different total), `--parallel 4` (fewer
requests at once if you hit rate limits). See all of them with
`python ganyu/generate_with_my_ai.py --help`.

### Exactly what your AI is asked

Each request has two parts, and both live at the top of
`generate_with_my_ai.py` (`SYSTEM_PROMPT`, `EXAMPLE_PROMPT`) where you can edit
them.

**1. System prompt, the same every time.** It contains:
- the job: "write ONE realistic chat between a person and Ganyu",
- the whole character sheet from `persona.py` (edit that file to change her),
- 6 rules: she always speaks as herself, never mentions AI or role-play; she
  knows nothing of the modern world; all wording is original; the person
  sounds like a real person; plain prose replies; output ONLY one JSON object
  in the exact format shown.

**2. Message, different every time.** For example:

```
Write conversation #17.

Topic: teasing her about being a goat, eating grass, or her horns
The person: a playful friend who loves teasing her
Exchanges: 2 (one exchange = one message from the person + one reply from Ganyu)
Length of each of Ganyu's replies: medium: 3-5 sentences
Extra twist: Ganyu gets a little flustered at some point

Output only the JSON object.
```

To see the full, exact text, run:

```bash
python ganyu/generate_with_my_ai.py --print-prompt
```

### Using a chat website instead (ChatGPT, Claude.ai, Gemini…)

Run `--print-prompt`, paste the **system prompt** as your first message (or into
"custom instructions"), then send this to get 10 at a time:

```
Write 10 different conversations following the rules above. Vary the topic,
the person's style, the length, and use 1-3 exchanges. Output exactly 10 lines,
one JSON object per line, and nothing else.
```

Copy the 10 lines into `my_data/conversations.jsonl` (one per line), and repeat
about 50 times. The checker below will catch any line that came out wrong.

## Steps

```bash
python ganyu/prepare_my_data.py --check
```

This checks your files for mistakes. It's free and doesn't call Tinker.

```bash
python ganyu/prepare_my_data.py
```

This builds `ganyu/data/my_train.jsonl`. It calls Tinker only for
`ganyu_lines.txt` and `questions.txt`.

```bash
python ganyu/train.py --mine-repeat 3
```

This trains on the generated data plus your data. Your data is counted 3 times,
so your writing has more influence. Other options:

- `python ganyu/train.py --only-mine` trains on your data alone (best with 200+ conversations).
- `python ganyu/train.py --no-mine` ignores your data.

```bash
python ganyu/compare.py
```

This tests the result.

## Tips for good data

- **Quality beats quantity, but you still need volume.** Around 200 conversations
  is a good minimum for a stable persona. Below that, mix in the generated data.
- **Variety matters.** Cover many situations: small talk, work, feelings,
  teasing, identity questions, modern-world questions.
- **The model copies what you show it.** If all your Ganyu lines are one
  sentence long, she will answer in one sentence. Long lines give long answers.
- **Never mention the character sheet or "role-play" in your data.** Write
  Ganyu's lines as Ganyu herself would say them.
