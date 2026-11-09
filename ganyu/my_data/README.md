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
