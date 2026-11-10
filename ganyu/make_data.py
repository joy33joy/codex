"""
STEP 1: Create training conversations.

How it works:
  1. The teacher model invents many different things a person might say to
     Ganyu, one batch per topic in persona.py.
  2. The teacher answers each one *while reading Ganyu's character sheet*.
     Some conversations get a second back-and-forth turn.
  3. We save the conversations WITHOUT the character sheet. The student later
     learns to answer like this with no sheet at all, which is what makes
     Ganyu its own identity instead of an instruction it follows.

Run:  python ganyu/make_data.py
"""

import asyncio
import json
import random
import re

import tinker
from tqdm.asyncio import tqdm_asyncio

from common import (DATA_FILE, TEACHER_MODEL, TEACHER_RENDERER, chat_once, get_renderer,
                    is_planning_notes)
from persona import (EDGE_QUESTIONS, FORBIDDEN_WORDS, IDENTITY_QUESTIONS, MODERN_QUESTIONS, PERSONA,
                     TOPICS)

MESSAGES_PER_TOPIC = 15  # 36 topics x 15 = 540 invented messages
ANSWERS_PER_IDENTITY_QUESTION = 3  # identity matters most, so sample it more
ANSWERS_PER_MODERN_QUESTION = 3  # the base model knows modern things; override hard
ANSWERS_PER_EDGE_QUESTION = 3  # weak spots found by redteam.py
FOLLOW_UP_CHANCE = 0.4  # share of conversations that get a second turn
MAX_PARALLEL = 32
SEED = 0


def looks_ok(reply: str) -> bool:
    low = reply.lower()
    return (bool(reply) and len(reply) < 2500 and not is_planning_notes(reply)
            and not any(w in low for w in FORBIDDEN_WORDS))


def parse_json_list(text: str) -> list[str]:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        items = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


async def main() -> None:
    random.seed(SEED)
    service = tinker.ServiceClient()
    teacher = await service.create_sampling_client(base_model=TEACHER_MODEL)
    renderer = get_renderer(TEACHER_MODEL, TEACHER_RENDERER)
    gate = asyncio.Semaphore(MAX_PARALLEL)

    async def ask(messages, **kw) -> str:
        async with gate:
            return await chat_once(teacher, renderer, messages, **kw)

    # --- 1. Invent user messages for every topic -------------------------
    async def invent(topic: str) -> list[str]:
        request = (
            f"Write {MESSAGES_PER_TOPIC} different messages that a person might send to "
            f"Ganyu (the half-qilin secretary of the Liyue Qixing) on this theme: {topic}.\n"
            "Vary the length, tone, and writing style: some casual, some formal, some "
            "with typos, some very short. Write only what the person says.\n"
            "Answer with a JSON list of strings and nothing else."
        )
        return parse_json_list(await ask([{"role": "user", "content": request}], temperature=1.0))

    print("Inventing user messages...")
    batches = await tqdm_asyncio.gather(*(invent(t) for t in TOPICS))
    questions = [q for batch in batches for q in batch if not is_planning_notes(q)]
    questions += IDENTITY_QUESTIONS * ANSWERS_PER_IDENTITY_QUESTION
    questions += MODERN_QUESTIONS * ANSWERS_PER_MODERN_QUESTION
    questions += EDGE_QUESTIONS * ANSWERS_PER_EDGE_QUESTION
    print(f"{len(questions)} user messages")

    # --- 2. Ganyu answers (teacher reads the character sheet) ------------
    def as_ganyu(history: list[dict]) -> list[dict]:
        return [{"role": "system", "content": PERSONA}, *history]

    async def make_conversation(question: str) -> list[dict] | None:
        convo = [{"role": "user", "content": question}]
        reply = await ask(as_ganyu(convo))
        if not looks_ok(reply):
            return None
        convo.append({"role": "assistant", "content": reply})

        if random.random() < FOLLOW_UP_CHANCE:
            transcript = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in convo)
            follow_up = await ask(
                [{"role": "user", "content": (
                    "Here is a chat between a person (USER) and Ganyu (ASSISTANT):\n\n"
                    f"{transcript}\n\nWrite the person's next message only, naturally "
                    "continuing the conversation. Sometimes push back, tease, or ask a "
                    "deeper question. Output only the message."
                )}],
                temperature=1.0,
                max_tokens=200,
            )
            if follow_up and not is_planning_notes(follow_up):
                second = await ask(as_ganyu([*convo, {"role": "user", "content": follow_up}]))
                if looks_ok(second):
                    convo += [{"role": "user", "content": follow_up},
                              {"role": "assistant", "content": second}]
        return convo

    print("Writing Ganyu's answers...")
    convos = await tqdm_asyncio.gather(*(make_conversation(q) for q in questions))
    kept = [c for c in convos if c]

    # --- 3. Save, with NO character sheet inside -------------------------
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DATA_FILE, "w") as f:
        for c in kept:
            f.write(json.dumps({"messages": c}, ensure_ascii=False) + "\n")
    print(f"Saved {len(kept)} conversations ({len(convos) - len(kept)} rejected) to {DATA_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
