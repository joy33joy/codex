"""
YOUR OWN DATA: turn the files in ganyu/my_data/ into training conversations.

Put any of these files in ganyu/my_data/ (all optional, mix as you like):

  conversations.jsonl  Full chats you wrote, one per line:
                       {"messages": [{"role": "user", "content": "..."},
                                     {"role": "assistant", "content": "..."}]}
  pairs.csv            A spreadsheet with two columns: user,ganyu
  ganyu_lines.txt      Just Ganyu's lines, one per line. The teacher writes a
                       user message that would naturally lead to each one.
  questions.txt        Just messages TO Ganyu, one per line. The teacher
                       answers each one in character (using persona.py).

Examples of every format are in ganyu/my_data/examples/.

Run:
  python ganyu/prepare_my_data.py --check   # only check your files, free, no Tinker
  python ganyu/prepare_my_data.py           # build ganyu/data/my_train.jsonl
"""

import argparse
import asyncio
import csv
import json
import sys

import tinker
from tqdm.asyncio import tqdm_asyncio

from common import (HERE, MY_DATA_FILE, TEACHER_MODEL, TEACHER_RENDERER, chat_once, get_renderer,
                    is_planning_notes)
from make_data import looks_ok
from persona import FORBIDDEN_WORDS, PERSONA

MY_DIR = HERE / "my_data"
MAX_PARALLEL = 32


def read_lines(name: str) -> list[str]:
    path = MY_DIR / name
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_conversations(problems: list[str]) -> list[list[dict]]:
    path = MY_DIR / "conversations.jsonl"
    if not path.exists():
        return []
    convos = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            msgs = json.loads(line)["messages"]
        except (json.JSONDecodeError, KeyError, TypeError):
            problems.append(f"conversations.jsonl line {n}: not valid JSON with a \"messages\" list")
            continue
        roles = [m.get("role") for m in msgs]
        if not msgs or any(r not in ("user", "assistant") for r in roles) or "assistant" not in roles:
            problems.append(f"conversations.jsonl line {n}: roles must be 'user'/'assistant' "
                            "and include at least one assistant (Ganyu) reply")
            continue
        if any(not str(m.get("content", "")).strip() for m in msgs):
            problems.append(f"conversations.jsonl line {n}: has an empty message")
            continue
        convos.append([{"role": m["role"], "content": m["content"].strip()} for m in msgs])
    return convos


def read_pairs(problems: list[str]) -> list[list[dict]]:
    path = MY_DIR / "pairs.csv"
    if not path.exists():
        return []
    convos = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        if "user" not in cols or "ganyu" not in cols:
            problems.append(f"pairs.csv: needs a header row with columns 'user' and 'ganyu' "
                            f"(found: {reader.fieldnames})")
            return []
        for n, row in enumerate(reader, 2):
            user, ganyu = (row[cols["user"]] or "").strip(), (row[cols["ganyu"]] or "").strip()
            if not user or not ganyu:
                problems.append(f"pairs.csv row {n}: empty 'user' or 'ganyu' cell")
                continue
            convos.append([{"role": "user", "content": user}, {"role": "assistant", "content": ganyu}])
    return convos


def warn_forbidden(convos: list[list[dict]], warnings: list[str]) -> None:
    """Your own lines are kept as written, but flag words that may break the illusion."""
    for c in convos:
        for m in c:
            if m["role"] == "assistant":
                hits = [w for w in FORBIDDEN_WORDS if w in m["content"].lower()]
                if hits:
                    warnings.append(f"Ganyu says {hits} in: \"{m['content'][:70]}...\"")


async def generate(lines: list[str], questions: list[str]) -> tuple[list[list[dict]], int]:
    service = tinker.ServiceClient()
    teacher = await service.create_sampling_client(base_model=TEACHER_MODEL)
    renderer = get_renderer(TEACHER_MODEL, TEACHER_RENDERER)
    gate = asyncio.Semaphore(MAX_PARALLEL)

    async def ask(messages, **kw) -> str:
        async with gate:
            return await chat_once(teacher, renderer, messages, **kw)

    async def lead_in(line: str) -> list[dict] | None:
        user = await ask([{"role": "user", "content": (
            "Ganyu (the half-qilin secretary of the Liyue Qixing) says this line in a chat:\n\n"
            f"\"{line}\"\n\nWrite the one message a person would most naturally have sent "
            "to Ganyu right before, so that her line is a fitting reply. "
            "Output only that message."
        )}], temperature=0.9, max_tokens=200)
        user = user.strip().strip('"')
        if not user or is_planning_notes(user):
            return None
        return [{"role": "user", "content": user}, {"role": "assistant", "content": line}]

    async def answer(question: str) -> list[dict] | None:
        reply = await ask([{"role": "system", "content": PERSONA}, {"role": "user", "content": question}])
        return [{"role": "user", "content": question}, {"role": "assistant", "content": reply}] \
            if looks_ok(reply) else None

    jobs = [lead_in(l) for l in lines] + [answer(q) for q in questions]
    results = await tqdm_asyncio.gather(*jobs)
    kept = [r for r in results if r]
    return kept, len(results) - len(kept)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="only check your files; no Tinker calls")
    args = parser.parse_args()

    problems: list[str] = []
    warnings: list[str] = []
    convos = read_conversations(problems) + read_pairs(problems)
    lines = read_lines("ganyu_lines.txt")
    questions = read_lines("questions.txt")
    warn_forbidden(convos + [[{"role": "assistant", "content": l}] for l in lines], warnings)

    print(f"Found in {MY_DIR}:")
    print(f"  ready-made conversations (conversations.jsonl + pairs.csv): {len(convos)}")
    print(f"  Ganyu lines needing a user message (ganyu_lines.txt):       {len(lines)}")
    print(f"  questions needing a Ganyu answer (questions.txt):           {len(questions)}")
    for p in problems:
        print(f"  PROBLEM (skipped): {p}")
    for w in warnings:
        print(f"  warning (kept):    {w}")

    if not (convos or lines or questions):
        print("\nNo data found. Copy a file from ganyu/my_data/examples/ into ganyu/my_data/ to start.")
        sys.exit(1)
    if args.check:
        print("\nCheck finished. Run again without --check to build the training file.")
        return

    rejected = 0
    if lines or questions:
        print(f"\nAsking the teacher model ({TEACHER_MODEL}) to fill in the missing halves...")
        generated, rejected = asyncio.run(generate(lines, questions))
        convos += generated

    MY_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MY_DATA_FILE, "w", encoding="utf-8") as f:
        for c in convos:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(convos)} conversations to {MY_DATA_FILE}"
          + (f" ({rejected} generated answers rejected by the filter)" if rejected else ""))
    print("Next: python ganyu/train.py   (see --help for mixing options)")


if __name__ == "__main__":
    main()
