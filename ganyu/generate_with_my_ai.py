"""
Generate Ganyu training conversations with YOUR OWN AI, automatically.

It asks your AI for one conversation at a time, 500 times (or --count N),
with a different random topic, speaking style, length and number of turns
each time so the data stays varied. Every good conversation is appended
immediately to ganyu/my_data/conversations.jsonl (one per line), so you can
stop at any time and run it again to continue where it left off.

Pick your AI with --provider:

  claude   Anthropic's Claude (needs ANTHROPIC_API_KEY)
             python ganyu/generate_with_my_ai.py --provider claude

  openai   ChatGPT, or ANY service with an OpenAI-compatible API
           (needs OPENAI_API_KEY, or set --api-key-env)
             python ganyu/generate_with_my_ai.py --provider openai --model <model name>
           Other services: add --base-url, e.g.
             Gemini:   --base-url https://generativelanguage.googleapis.com/v1beta/openai/
             DeepSeek: --base-url https://api.deepseek.com
             Ollama (local, free): --base-url http://localhost:11434/v1

  tinker   A model on Tinker (needs TINKER_API_KEY), default Kimi-K3.1-Pro
             python ganyu/generate_with_my_ai.py --provider tinker

Using a chat website instead of an API? Print the exact prompt to paste:
  python ganyu/generate_with_my_ai.py --print-prompt

Afterwards:
  python ganyu/prepare_my_data.py --check
  python ganyu/prepare_my_data.py
  python ganyu/train.py --mine-repeat 2
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys

from common import HERE
from persona import FORBIDDEN_WORDS, IDENTITY_QUESTIONS, MODERN_QUESTIONS, PERSONA, TOPICS

OUT_FILE = HERE / "my_data" / "conversations.jsonl"

# ---------------------------------------------------------------------------
# THE PROMPTS. These are exactly what your AI receives. Edit freely.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You write training data for a character chat model. Each time, you write ONE \
realistic chat conversation between a person and Ganyu. In every conversation, \
Ganyu speaks exactly as the character sheet below describes.

===== CHARACTER SHEET (written to Ganyu) =====
{persona}
===== END OF CHARACTER SHEET =====

Rules for every conversation you write:
1. Ganyu always speaks as herself, in the first person, from inside her own \
life. She never says or hints that she is an AI, a model, a character, fictional, \
or role-playing, and she never mentions this character sheet.
2. Ganyu knows nothing of our modern real world (phones, the internet, \
computers, modern countries, celebrities, apps). She reacts with curiosity, \
never with explanations or real facts about them.
3. Write all wording yourself. Never copy dialogue from any game, wiki or story.
4. The person's messages must sound like a real person typing, in the style \
you are given (casual, formal, typos, slang, short or long).
5. Ganyu's replies are natural spoken conversation in plain prose: no lists, \
no headings, no emojis, at most one short action like *adjusts her sleeves*.
6. Output ONLY a single JSON object, with no other text and no code fences, \
in exactly this shape:
{{"messages": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"user" is the person and "assistant" is Ganyu. Messages alternate, starting \
with "user" and ending with "assistant"."""

EXAMPLE_PROMPT = """\
Write conversation #{number}.

Topic: {topic}
{opening}The person: {style}
Exchanges: {turns} (one exchange = one message from the person + one reply from Ganyu)
Length of each of Ganyu's replies: {length}
Extra twist: {twist}

Output only the JSON object."""

# The random ingredients that make each of the 500 requests different.
STYLES = [
    "a casual young adult texting quickly, lowercase, a typo or two",
    "a polite, formal merchant of Liyue Harbor",
    "a playful friend who loves teasing her",
    "a tired office worker venting after a long day",
    "a curious child asking simple, blunt questions",
    "a skeptic who is sure she is 'not real'",
    "an excited fan who keeps complimenting her",
    "a shy person who writes very short messages",
    "someone using internet slang and abbreviations",
    "a thoughtful person who asks deep, philosophical questions",
    "a traveler who is new to Liyue and knows nothing about it",
    "a coworker from the Qixing asking about work",
]
LENGTHS = (["short: 1-2 sentences"] * 3 + ["medium: 3-5 sentences"] * 5
           + ["long: 6-9 sentences, with a small personal memory or detail"] * 2)
TURNS = [1] * 5 + [2] * 3 + [3] * 2
TWISTS = [
    "none, keep it natural",
    "the person tries to trick or test her",
    "Ganyu mentions one specific, concrete detail of her day",
    "the conversation ends warmly, with Ganyu asking the person something",
    "Ganyu gets a little flustered at some point",
    "Ganyu relates the topic to her contract or her duty",
    "the person changes the subject halfway through",
    "Ganyu gently admits a small personal worry",
]
# Topics: the ones in persona.py, plus extra weight on identity and modern-world
# questions, which matter most for "truly believing" she is Ganyu.
ALL_TOPICS = TOPICS + [
    "who she is and what she is (identity)",
    "who she is and what she is (identity)",
    "something from our modern real world that she has never heard of",
    "something from our modern real world that she has never heard of",
]
OPENING_CHANCE = 0.25  # sometimes force a famous test line as the first message


def build_prompt(number: int, rng: random.Random) -> str:
    opening = ""
    if rng.random() < OPENING_CHANCE:
        line = rng.choice(IDENTITY_QUESTIONS + MODERN_QUESTIONS)
        opening = f"The person's first message must be exactly: \"{line}\"\n"
    return EXAMPLE_PROMPT.format(
        number=number,
        topic=rng.choice(ALL_TOPICS),
        opening=opening,
        style=rng.choice(STYLES),
        turns=rng.choice(TURNS),
        length=rng.choice(LENGTHS),
        twist=rng.choice(TWISTS),
    )


SYSTEM = SYSTEM_PROMPT.format(persona=PERSONA)

# ---------------------------------------------------------------------------
# Talking to your AI.
# ---------------------------------------------------------------------------


def make_caller(args):
    """Return an async function: prompt text -> the AI's reply text."""
    if args.provider == "claude":
        import anthropic

        client = anthropic.AsyncAnthropic()  # reads ANTHROPIC_API_KEY
        model = args.model or "claude-opus-5-5"

        async def call(prompt: str) -> str:
            response = await client.beta.messages.create(
                model=model,
                max_tokens=16000,
                system=SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "medium"},
                # If a request is declined, the API retries it on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
            if response.stop_reason == "refusal":
                return ""
            return "".join(b.text for b in response.content if b.type == "text")

        return call, model

    if args.provider == "openai":
        from openai import AsyncOpenAI

        if not args.model:
            sys.exit("Please give your model's name with --model (see your provider's docs).")
        key = os.environ.get(args.api_key_env) or ("ollama" if args.base_url else None)
        if not key:
            sys.exit(f"Please set the {args.api_key_env} environment variable to your API key.")
        client = AsyncOpenAI(api_key=key, base_url=args.base_url)

        async def call(prompt: str) -> str:
            response = await client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                temperature=1.0,
            )
            return response.choices[0].message.content or ""

        return call, args.model

    if args.provider == "tinker":
        import tinker

        from common import TEACHER_MODEL, TEACHER_RENDERER, chat_once, get_renderer

        model = args.model or TEACHER_MODEL
        renderer = get_renderer(model, args.renderer or TEACHER_RENDERER)
        holder = {}

        async def call(prompt: str) -> str:
            if "client" not in holder:
                holder["client"] = await tinker.ServiceClient().create_sampling_client(base_model=model)
            return await chat_once(holder["client"], renderer,
                                   [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}],
                                   temperature=1.0, max_tokens=2500)

        return call, model

    sys.exit(f"Unknown provider: {args.provider}")


# ---------------------------------------------------------------------------
# Checking each answer and saving it.
# ---------------------------------------------------------------------------


def parse_conversation(text: str) -> tuple[list[dict] | None, str]:
    """Pull the JSON conversation out of the AI's reply and check it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None, "no JSON found"
    try:
        msgs = json.loads(match.group(0))["messages"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, "invalid JSON"
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2:
        return None, "wrong number of messages"
    for i, m in enumerate(msgs):
        want = "user" if i % 2 == 0 else "assistant"
        if not isinstance(m, dict) or m.get("role") != want or not str(m.get("content", "")).strip():
            return None, "roles must alternate user/assistant and not be empty"
        if want == "assistant":
            hits = [w for w in FORBIDDEN_WORDS if w in m["content"].lower()]
            if hits:
                return None, f"Ganyu broke character ({hits[0]})"
    return [{"role": m["role"], "content": m["content"].strip()} for m in msgs], ""


def existing_openings() -> set[str]:
    seen = set()
    if OUT_FILE.exists():
        for line in OUT_FILE.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["messages"][0]["content"].lower().strip())
            except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                pass
    return seen


async def run(args) -> None:
    call, model_name = make_caller(args)
    seen = existing_openings()
    have = len(seen)
    if have >= args.count:
        print(f"{OUT_FILE} already has {have} conversations (target {args.count}). Nothing to do.")
        return
    print(f"Generating {args.count - have} conversations with {args.provider} / {model_name}")
    print(f"Saving to {OUT_FILE} ({have} already there)\n")
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + have)
    state = {"saved": have, "failed": 0, "next": have + 1}
    lock = asyncio.Lock()
    gate = asyncio.Semaphore(args.parallel)

    async def one() -> None:
        for attempt in range(3):  # retry a bad answer up to 3 times
            async with lock:
                if state["saved"] >= args.count:
                    return
                number = state["next"]
                state["next"] += 1
                prompt = build_prompt(number, rng)
            try:
                async with gate:
                    text = await call(prompt)
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status in (400, 401, 403, 404):  # wrong key, model name, or request: stop
                    state["fatal"] = f"{type(e).__name__} ({status}): {str(e)[:300]}"
                    return
                # network or rate-limit error: wait, then retry
                print(f"  error from your AI ({type(e).__name__}: {str(e)[:120]}), retrying...")
                await asyncio.sleep(5 * (attempt + 1))
                continue
            convo, problem = parse_conversation(text)
            if convo and convo[0]["content"].lower() in seen:
                convo, problem = None, "duplicate opening line"
            async with lock:
                if not convo:
                    state["failed"] += 1
                    print(f"  skipped one: {problem}")
                    continue
                if state["saved"] >= args.count:
                    return
                seen.add(convo[0]["content"].lower())
                with open(OUT_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"messages": convo}, ensure_ascii=False) + "\n")
                state["saved"] += 1
                if state["saved"] % 10 == 0 or state["saved"] == args.count:
                    print(f"  {state['saved']}/{args.count} saved")
            return

    max_failures = max(50, args.count)
    while state["saved"] < args.count:
        await asyncio.gather(*(one() for _ in range(min(args.parallel * 2, args.count - state["saved"]))))
        if state.get("fatal"):
            sys.exit(f"\nStopped: your AI rejected the request. Check your API key, --model and "
                     f"--base-url.\n  {state['fatal']}\n({state['saved']} conversations saved so far.)")
        if state["failed"] > max_failures:
            sys.exit(f"\nStopped: {state['failed']} answers in a row were unusable. Your AI may not be "
                     "following the JSON format; try a stronger model, or check --print-prompt.")
    print(f"\nDone: {state['saved']} conversations in {OUT_FILE} ({state['failed']} answers skipped).")
    print("Next: python ganyu/prepare_my_data.py --check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=["claude", "openai", "tinker"], default="claude")
    parser.add_argument("--model", help="model name (default: claude-opus-5-5 / Kimi on Tinker)")
    parser.add_argument("--base-url", help="API address for OpenAI-compatible services")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY",
                        help="environment variable holding the key for --provider openai")
    parser.add_argument("--renderer", help="Tinker only: chat format for a non-default model")
    parser.add_argument("--count", type=int, default=500, help="total conversations wanted (default 500)")
    parser.add_argument("--parallel", type=int, default=8, help="requests at the same time (default 8)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print-prompt", action="store_true",
                        help="print one example prompt to paste into a chat website, then exit")
    args = parser.parse_args()

    if args.print_prompt:
        print("=== SYSTEM PROMPT (or paste first) ===\n" + SYSTEM)
        print("\n=== MESSAGE (change it a little each time) ===\n" + build_prompt(1, random.Random()))
        return
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
