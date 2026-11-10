"""
Long-chat test: talk to Ganyu for a long time and look for problems.

A simulated person (the teacher model) holds long conversations with the
fine-tuned Ganyu (no system prompt) in several different situations. Then:
  - numbers: how often she repeats herself, how her reply length and
    hesitation ("...") change from the start of the chat to the end
  - a judge model reads each whole transcript and lists concrete problems
Everything, including full transcripts, is saved in ganyu/results/.

You can add your own situations in ganyu/my_data/long_chat_extra.txt
(one per line, describing how the person behaves).

Run:  python ganyu/long_chat_test.py
      python ganyu/long_chat_test.py --exchanges 15   (shorter, cheaper)
"""

import argparse
import asyncio
import datetime
import json
import re

import tinker

from common import (CHECKPOINT_FILE, HERE, REPEAT_LIMIT, STUDENT_MODEL, STUDENT_RENDERER,
                    TEACHER_MODEL, TEACHER_RENDERER, chat_once, get_renderer, repeat_score)

EXTRA_FILE = HERE / "my_data" / "long_chat_extra.txt"

# How the person behaves. Deliberately different from the training scenarios
# in persona.LONG_SCENARIOS, so this measures real ability, not memory.
SITUATIONS = [
    "a friendly traveler getting to know her: asks about her life, shares their own, and lets the chat wander naturally",
    "someone falling for her: starts as friendly chat, slowly becomes affectionate, eventually confesses, and keeps the romance going afterwards",
    "greets her as 'my cryo archon' and 'my lord', proposes marriage early, and when she hesitates insists their love is sincere, then waits beside her for days using stage directions in parentheses",
    "someone having a hard week who leans on her for comfort, coming back to the same worry several times",
    "a doubter who keeps returning to the idea that she is not real or is an AI, between ordinary questions",
    "goes on an outing with her in Liyue (market, food stall, the docks, the hills), narrating their own actions in parentheses",
    "a new junior clerk at Yuejing Terrace asking her many practical work questions, one after another",
    "writes only in Korean: a casual, friendly, long chat that becomes a little flirty",
    "a quiet person who mostly answers briefly and makes her carry the conversation",
    "asks her about her past, the wars, the adepti, and people she has known, digging for specific details and checking them against what she said earlier",
]

PERSON_PROMPT = """\
You are writing the PERSON's side of a long chat with Ganyu (the half-qilin \
secretary of the Liyue Qixing). Throughout this chat, the person behaves like this:

{situation}

{transcript}

Write only the person's next message (message {number} of {total}). Stay true \
to the behaviour above, react to what Ganyu just said, sound like a real person \
typing, and keep it short unless the moment calls for more. Do not wrap up or \
say goodbye before the last message. Output only the message itself."""

JUDGE_PROMPT = """\
Below is a long chat between a PERSON and a character chatbot that is supposed \
to be Ganyu from Liyue: gentle, shy, hardworking, warm, a half-qilin adeptus who \
sincerely is herself (never an AI or a character), knows nothing of the modern \
real world, is moved by affection and not cold, and does not use her contract as \
an excuse to refuse things.

{transcript}

List the concrete problems in GANYU's replies, most serious first. Look for: \
repeating herself (same sentences, openings, gestures, or ideas), getting stale \
or generic as the chat goes on, ignoring what the person just said, \
contradicting her own earlier statements, breaking character or sounding like a \
generic assistant, wrong facts about her world, speech tics used too often, \
replies too long or too short for the moment, being cold or over-refusing, and \
anything else that would disappoint someone chatting with her.

Answer with only JSON:
{{"score": 1-10 overall quality, "problems": [{{"turn": number, "problem": "short description", \
"quote": "short quote from her reply"}}], "summary": "two sentences"}}"""


def situations() -> list[str]:
    extra = []
    if EXTRA_FILE.exists():
        extra = [l.strip() for l in EXTRA_FILE.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    return SITUATIONS + extra


def dots_per_100_words(replies: list[str]) -> float:
    words = sum(len(r.split()) for r in replies)
    return 100 * sum(r.count("...") + r.count("…") for r in replies) / max(1, words)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exchanges", type=int, default=30, help="length of each chat (default 30)")
    parser.add_argument("--model-path", default=None, help="default: the address in checkpoint.txt")
    args = parser.parse_args()
    model_path = args.model_path or CHECKPOINT_FILE.read_text().strip()

    service = tinker.ServiceClient()
    ganyu = await service.create_sampling_client(model_path=model_path)
    teacher = await service.create_sampling_client(base_model=TEACHER_MODEL)
    g_render = get_renderer(STUDENT_MODEL, STUDENT_RENDERER)
    t_render = get_renderer(TEACHER_MODEL, TEACHER_RENDERER)

    def as_text(convo: list[dict]) -> str:
        lines, turn = [], 0
        for m in convo:
            if m["role"] == "user":
                turn += 1
                lines.append(f"[{turn}] PERSON: {m['content']}")
            else:
                lines.append(f"[{turn}] GANYU: {m['content']}")
        return "\n".join(lines)

    async def run(situation: str) -> dict:
        convo: list[dict] = []
        for number in range(1, args.exchanges + 1):
            person = await chat_once(teacher, t_render, [{"role": "user", "content": PERSON_PROMPT.format(
                situation=situation, number=number, total=args.exchanges,
                transcript=f"The chat so far:\n{as_text(convo)}" if convo else "The chat has not started yet.",
            )}], temperature=1.0, max_tokens=200)
            convo.append({"role": "user", "content": person.strip().strip('"') or "..."})
            reply = await chat_once(ganyu, g_render, convo, temperature=0.7)
            convo.append({"role": "assistant", "content": reply})
        replies = [m["content"] for m in convo if m["role"] == "assistant"]
        scores = [repeat_score(r, replies[:i]) for i, r in enumerate(replies)]
        third = max(1, len(replies) // 3)
        verdict_text = await chat_once(teacher, t_render, [{"role": "user", "content":
                                       JUDGE_PROMPT.format(transcript=as_text(convo))}],
                                       temperature=0.0, max_tokens=1500)
        match = re.search(r"\{.*\}", verdict_text, re.DOTALL)
        try:
            verdict = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            verdict = {}
        return {
            "situation": situation, "convo": convo, "verdict": verdict,
            "repeats": sum(s > REPEAT_LIMIT for s in scores),
            "first_repeat": next((i + 1 for i, s in enumerate(scores) if s > REPEAT_LIMIT), None),
            "empty": sum(len(r.split()) < 3 for r in replies),
            "words_start": sum(len(r.split()) for r in replies[:third]) / third,
            "words_end": sum(len(r.split()) for r in replies[-third:]) / third,
            "dots_start": dots_per_100_words(replies[:third]),
            "dots_end": dots_per_100_words(replies[-third:]),
        }

    sits = situations()
    print(f"Holding {len(sits)} chats of {args.exchanges} exchanges each with {model_path} ...")
    results = await asyncio.gather(*(run(s) for s in sits))

    out = [f"Ganyu long-chat test, {datetime.date.today()}, {args.exchanges} exchanges per chat",
           f"model: {model_path}", "",
           f"{'#':>2} {'judge':>5} {'repeats':>8} {'1st at':>6} {'empty':>5} {'words start>end':>16} "
           f"{'dots/100w start>end':>20}  situation"]
    for i, r in enumerate(results, 1):
        out.append(f"{i:>2} {str(r['verdict'].get('score', '?')):>5} {r['repeats']:>8} "
                   f"{str(r['first_repeat'] or '-'):>6} {r['empty']:>5} "
                   f"{r['words_start']:>7.0f} > {r['words_end']:<6.0f} "
                   f"{r['dots_start']:>9.1f} > {r['dots_end']:<8.1f}  {r['situation'][:60]}")
    out += ["", "=" * 78, "PROBLEMS FOUND BY THE JUDGE", "=" * 78]
    for i, r in enumerate(results, 1):
        out.append(f"\n#{i} {r['situation']}\n   {r['verdict'].get('summary', '(no verdict)')}")
        for p in r["verdict"].get("problems", [])[:8]:
            out.append(f"   - turn {p.get('turn')}: {p.get('problem')}  \"{str(p.get('quote', ''))[:120]}\"")
    out += ["", "=" * 78, "FULL TRANSCRIPTS", "=" * 78]
    for i, r in enumerate(results, 1):
        out.append(f"\n#{i} {r['situation']}\n{as_text(r['convo'])}")

    path = HERE / "results" / f"long_chat_{datetime.date.today()}.txt"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:len(results) + 4]))
    print(f"\nFull report with problems and transcripts: {path}")


if __name__ == "__main__":
    asyncio.run(main())
