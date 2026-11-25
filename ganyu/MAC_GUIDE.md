# Run Ganyu on your Mac (abliterated Qwen3.8-27B + Ganyu adapter)

This runs Ganyu entirely on your Mac with **llama.cpp**: the abliterated
Qwen3.8-27B, compressed to 3 bits so it fits in 24 GB of memory, with the
Ganyu adapter attached. After setup, chatting needs no internet and no Tinker.

> **Why not LM Studio?** LM Studio can't attach a separate adapter file; it
> only loads complete models. llama.cpp can, and it comes with its own chat
> page in the browser.

**You need:**
- A Mac with Apple silicon and at least 24 GB of memory
- About **20 GB of free disk space**
- Your **Tinker API key**, used once to download the adapter and again for the tests

Every command below goes into the **Terminal** app (Applications → Utilities →
Terminal). Paste one block at a time and press Enter.

---

## Part 1: Install the tools (once)

**1.** Install Homebrew if you don't have it (it's the standard Mac package installer):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

When it finishes, it prints two or three commands under "Next steps". Run those too.

**2.** Install llama.cpp, Git and Python:

```bash
brew install llama.cpp git python@3.13
```

---

## Part 2: Get the project and build the adapter file (once)

**3.** Download this project and the llama.cpp source (the source holds the
converter script):

```bash
cd ~ && git clone https://github.com/joy33joy/codex.git && cd codex && git checkout claude/lucid-fermi-78ksdx
```

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp-src
```

**4.** Create a Python environment and install the packages (takes a few minutes):

```bash
cd ~/codex && python3.13 -m venv .venv && source .venv/bin/activate && pip install -r ganyu/requirements.txt gguf sentencepiece
```

**5.** Give Python your Tinker key for this Terminal window. Replace the
example with your real key:

```bash
export TINKER_API_KEY="tml-your-key-here"
```

**6.** Build the adapter file. This downloads the adapter from Tinker (about
1 GB) and converts it to the format llama.cpp uses:

```bash
python ganyu/export_adapter.py --llama-cpp ~/llama.cpp-src
```

It should end with `Done: .../ganyu/export/ganyu-lora-f16.gguf (546 MB)`.

**7.** Download the compressed abliterated model (13.5 GB; this takes a while).
If it stops partway, run the same command again and it resumes:

```bash
mkdir -p ~/ganyu-models && curl -L -C - -o ~/ganyu-models/Huihui-Qwen3.8-27B-abliterated.Q3_K_M.gguf https://huggingface.co/mradermacher/Huihui-Qwen3.8-27B-abliterated-GGUF/resolve/main/Huihui-Qwen3.8-27B-abliterated.Q3_K_M.gguf
```

---

## Part 3: Chat with her (every time)

**8.** Quit memory-heavy apps (browsers with many tabs, games, video editors), then start Ganyu:

```bash
llama-server -m ~/ganyu-models/Huihui-Qwen3.8-27B-abliterated.Q3_K_M.gguf --lora ~/codex/ganyu/export/ganyu-lora-f16.gguf -ngl 99 -c 8192 --cache-ram 0 --reasoning off --port 8080
```

Wait until it prints a line ending in `listening on http://127.0.0.1:8080`. The
first start takes a minute or two.

**9.** Open **http://localhost:8080** in your browser and start chatting.
Leave the system prompt empty: her identity comes from the adapter.

**10.** To stop her, press **Ctrl + C** in that Terminal window (or close it).
That frees the memory.

What each part of the step 8 command does:

| Part | Meaning |
|---|---|
| `-m …Q3_K_M.gguf` | The compressed abliterated model |
| `--lora …ganyu-lora-f16.gguf` | Attach the Ganyu adapter |
| `-ngl 99` | Run everything on the Mac's graphics chip (much faster) |
| `-c 8192` | Remember up to about 6,000 words of conversation |
| `--cache-ram 0` | Turn off llama-server's extra conversation cache (up to 8 GB by default), which a 24 GB Mac can't spare |
| `--reasoning off` | Thinking off, the way she was trained |

---

## Part 4: Test whether her personality survived (optional)

These are the same tests used on Tinker, now pointed at your Mac. Keep step 8
running, then open a **second** Terminal window and run:

```bash
cd ~/codex && source .venv/bin/activate && export TINKER_API_KEY="tml-your-key-here"
```

**Side by side with Ganyu on Tinker** (10 questions, 3 answers each):

```bash
python ganyu/compare.py --local http://localhost:8080 --samples 3 --save
```

**Robustness test** (32 tricky prompts, 3 tries each; scored by a judge on Tinker):

```bash
python ganyu/redteam.py --local http://localhost:8080
```

**Long-chat test** (10 conversations; 15 exchanges each keeps it shorter):

```bash
python ganyu/long_chat_test.py --local http://localhost:8080 --exchanges 15
```

Each test saves a report in `ganyu/results/`. Compare it with the Tinker
results already there (`redteam_v5_summary.txt`, `long_chat_after_round2.txt`).
Only Ganyu's replies come from your Mac. The simulated person and the judge
run on Tinker, so the tests use a few Tinker credits.

---

## If something goes wrong

| Problem | Fix |
|---|---|
| `failed to allocate` or out of memory at step 8 | Close other apps. Or use `-c 4096` instead of `-c 8192`. If it still fails, let the graphics chip use more of the memory (resets on restart): `sudo sysctl iogpu.wired_limit_mb=20480` |
| Very slow replies | Check `-ngl 99` is in the command, and that no other big app is open |
| She answers like a generic assistant | The adapter didn't load. Open http://localhost:8080/lora-adapters: it should list `ganyu-lora-f16.gguf` with `"scale":1.0`. If not, check the `--lora` path in step 8 |
| She writes `<think>` blocks | `--reasoning off` is missing from the step 8 command |
| `export_adapter.py` says the key is invalid | Re-run step 5 with your real Tinker key, in the same Terminal window |
| Download in step 7 stopped | Run the same command again; `-C -` resumes it |

## A smaller or sharper model

The 3-bit file (Q3_K_M, 13.5 GB) is the safe size for 24 GB. If it runs
comfortably, the 4-bit file is sharper but tighter on memory (16.8 GB). Replace
`Q3_K_M` with `Q4_K_M` in steps 7 and 8, and use `-c 4096`.
