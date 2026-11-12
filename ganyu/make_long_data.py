"""
STEP 1b: Create LONG training conversations (6-12 exchanges).

Why: make_data.py only makes short chats (1-2 exchanges). A model that has
never seen a long conversation drifts after a few turns and can start
copying its own earlier replies. These long chats teach Ganyu to keep every
reply fresh, especially when the person repeats themselves, goes in circles,
or lets time pass.

How it works, one conversation at a time:
  1. Pick a scenario from persona.LONG_SCENARIOS (how the person behaves).
  2. The teacher plays the PERSON: writes their next message for the scenario.
  3. The teacher plays GANYU (reading the character sheet): writes her reply.
     A reply that repeats her earlier ones is rejected and written again.
  4. Repeat 2-3 until the chat is 6-12 exchanges long.

Run:  python ganyu/make_long_data.py            (adds to data/ganyu_long.jsonl)
      python ganyu/make_long_data.py --count 10 (a small test)
"""

import argparse
import asyncio
import json
import random
import re

import tinker
from tqdm.asyncio import tqdm_asyncio

from common import (LONG_DATA_FILE, REPEAT_LIMIT, TEACHER_MODEL, TEACHER_RENDERER, calm, chat_once,
                    get_renderer, is_planning_notes, repeat_score)
from make_data import looks_ok
from persona import LONG_SCENARIOS, PERSONA

CONVERSATIONS_PER_SCENARIO = 4  # 22 scenarios x 4 = 88 long conversations
MIN_EXCHANGES, MAX_EXCHANGES = 6, 12
MAX_PARALLEL = 24
SEED = 0

PERSON_PROMPT = """\
You are writing the PERSON's side of a chat with Ganyu (the half-qilin \
secretary of the Liyue Qixing). Throughout this chat, the person behaves like this:

{scenario}

{transcript}

Write only the person's next message (message {number} of about {total}). Stay \
true to the behaviour above, sound like a real person typing, and keep it short \
unless the behaviour calls for more. Output only the message itself."""


def narrates(reply: str) -> bool:
    """True if Ganyu describes herself in the third person or piles up actions."""
    actions = re.findall(r"\*[^*\n]+\*|\([^()\n]+\)", reply)
    third_person = any(re.search(r"\b(she|her|hers|ganyu|ganyu's)\b", a.lower()) for a in actions)
    return third_person or len(actions) > 2


async def main() -> None:
    parser = argparse.ArgumentParser(description="Create long Ganyu training conversations.")
    parser.add_argument("--count", type=int, default=len(LONG_SCENARIOS) * CONVERSATIONS_PER_SCENARIO)
    parser.add_argument("--min", type=int, default=MIN_EXCHANGES, help="shortest chat, in exchanges")
    parser.add_argument("--max", type=int, default=MAX_EXCHANGES, help="longest chat, in exchanges")
    parser.add_argument("--append", action="store_true", help="add to the existing file instead of replacing it")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    teacher = await tinker.ServiceClient().create_sampling_client(base_model=TEACHER_MODEL)
    renderer = get_renderer(TEACHER_MODEL, TEACHER_RENDERER)
    gate = asyncio.Semaphore(MAX_PARALLEL)

    async def ask(messages, **kw) -> str:
        async with gate:
            return await chat_once(teacher, renderer, messages, **kw)

    async def make_conversation(scenario: str, exchanges: int) -> list[dict]:
        convo: list[dict] = []
        for number in range(1, exchanges + 1):
            transcript = "\n".join(
                f"{'PERSON' if m['role'] == 'user' else 'GANYU'}: {m['content']}" for m in convo)
            person = ""
            for _ in range(3):  # the teacher sometimes writes planning notes instead: ask again
                candidate = await ask([{"role": "user", "content": PERSON_PROMPT.format(
                    scenario=scenario, number=number, total=exchanges,
                    transcript=f"The chat so far:\n{transcript}" if convo else "The chat has not started yet.",
                )}], temperature=1.0, max_tokens=300)
                candidate = candidate.strip().strip('"')
                if candidate and len(candidate) < 600 and not is_planning_notes(candidate):
                    person = candidate
                    break
            if not person:
                break
            history = [*convo, {"role": "user", "content": person}]
            earlier = [m["content"] for m in convo if m["role"] == "assistant"]
            reply = ""
            for _ in range(3):  # reject replies that break character or repeat her earlier ones
                candidate = calm(await ask([{"role": "system", "content": PERSONA}, *history]))
                if (looks_ok(candidate) and not narrates(candidate)
                        and repeat_score(candidate, earlier) <= REPEAT_LIMIT):
                    reply = candidate
                    break
            if not reply:
                break
            convo = [*history, {"role": "assistant", "content": reply}]
        return convo

    jobs = [make_conversation(LONG_SCENARIOS[i % len(LONG_SCENARIOS)],
                              rng.randint(args.min, args.max))
            for i in range(args.count)]
    print(f"Writing {len(jobs)} long conversations with {TEACHER_MODEL}...")
    convos = await tqdm_asyncio.gather(*jobs)
    kept = [c for c in convos if len(c) >= 2 * 4]  # keep chats that reached at least 4 exchanges

    LONG_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LONG_DATA_FILE, "a" if args.append else "w", encoding="utf-8") as f:
        for c in kept:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    replies = sum(len(c) // 2 for c in kept)
    print(f"Saved {len(kept)} long conversations ({replies} Ganyu replies, "
          f"{len(convos) - len(kept)} dropped) to {LONG_DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
