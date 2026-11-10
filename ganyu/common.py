"""Shared settings used by all steps."""

import difflib
import re
from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# The "teacher" writes the training data (it reads the character sheet).
# Kimi-K3.1-Pro was the most natural Ganyu in a side-by-side test.
TEACHER_MODEL = "moonshotai/Kimi-K3.1-Pro:peft:262144"
TEACHER_RENDERER = "kimi_k31_pro_disable_thinking"

# The "student" is the model we fine-tune with LoRA. It becomes Ganyu.
STUDENT_MODEL = "Qwen/Qwen3.8-27B"
STUDENT_RENDERER = "qwen3_8_disable_thinking"  # hidden "thinking" off, so it just talks

HERE = Path(__file__).parent
DATA_FILE = HERE / "data" / "ganyu_train.jsonl"  # made by make_data.py
LONG_DATA_FILE = HERE / "data" / "ganyu_long.jsonl"  # made by make_long_data.py (long chats)
MY_DATA_FILE = HERE / "data" / "my_train.jsonl"  # made by prepare_my_data.py from your files
CHECKPOINT_FILE = HERE / "checkpoint.txt"  # where the trained model's address is saved


def get_renderer(model_name: str, renderer_name: str) -> renderers.Renderer:
    """How chat messages are turned into tokens for a given model."""
    tokenizer = get_tokenizer(model_name.split(":")[0])  # drop Tinker suffixes like ":peft:262144"
    return renderers.get_renderer(renderer_name, tokenizer)


async def chat_once(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    messages: list[dict],
    temperature: float = 0.8,
    max_tokens: int = 600,
) -> str:
    """Send a conversation to a model on Tinker and return its reply as text."""
    prompt = renderer.build_generation_prompt(messages)
    params = tinker.SamplingParams(
        max_tokens=max_tokens, temperature=temperature, stop=renderer.get_stop_sequences()
    )
    result = await sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=1)
    message, _ = renderer.parse_response(result.sequences[0].tokens)
    return renderers.get_text_content(message).strip()


# ---------------------------------------------------------------------------
# Repetition score: used when BUILDING training data (to reject replies that
# repeat earlier ones) and by the tests (to measure repetition). It is not
# used while chatting: the model itself should not repeat.
# ---------------------------------------------------------------------------

REPEAT_LIMIT = 0.5  # above this, a reply counts as a repeat


def repeat_score(reply: str, earlier_replies: list[str]) -> float:
    """0 = completely new, 1 = a copy of something Ganyu already said."""
    if not earlier_replies:
        return 0.0
    whole = max(difflib.SequenceMatcher(None, reply, e).ratio() for e in earlier_replies)
    # Also catch reusing whole sentences inside an otherwise new reply.
    said = {s.strip().lower() for e in earlier_replies for s in re.split(r"(?<=[.!?…])\s+", e)}
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?…])\s+", reply) if len(s.split()) >= 6]
    reused = sum(s in said for s in sentences) / len(sentences) if sentences else 0.0
    return max(whole, reused)



# ---------------------------------------------------------------------------
# Planning-note filter. The teacher model sometimes writes its own planning
# ("The user wants...", "I'll respond as Ganyu...") instead of the message it
# was asked for. If that gets into the training data, Ganyu learns to talk
# like that. Every teacher output is checked with this before it is kept.
# ---------------------------------------------------------------------------

_VERB = r"(respond|reply|write|produce|output|continue|play|react|draft|generate|analy[sz]e|consider)"
_NOTES_START = re.compile(
    r"^\W*(the user\b|user's message|user says|user wants|user asks|i see the user|i've compiled|"
    r"i (need|have) (to )?" + _VERB + r"|i('ll| will| should| must) " + _VERB + r"|i produce|"
    r"let me " + _VERB + r"|we need to " + _VERB + r"|should " + _VERB + r"|#+ ?task|task context|"
    r"incorrect\. please|okay,? (so )?the user|analysis:|draft:)", re.I)
_NOTES_ANYWHERE = re.compile(
    r"(the user wants|the user is|user would like|respond as ganyu|reply as ganyu|continue as ganyu|"
    r"roleplaying as ganyu|play the person|person's next message|next message only|output only|"
    r"\b\d-\d sentences|in first person,|stay in character|no analysis|message \d+ of \d+)", re.I)


def is_planning_notes(text: str) -> bool:
    """True if the text is the teacher's planning, not a real chat message."""
    return bool(_NOTES_START.search(text) or _NOTES_ANYWHERE.search(text))


# ---------------------------------------------------------------------------
# Calm-down filter. The teacher over-uses hesitation ("I... I...", "M-my",
# rows of "..."), and the student copies it. Ganyu is shy, not broken: this
# keeps at most two hesitations per reply and smooths out the rest.
# ---------------------------------------------------------------------------

MAX_HESITATIONS = 2
_ELLIPSIS = re.compile(r"\.\.\.|…")


def calm(reply: str) -> str:
    """Reduce stammering and trailing dots in one of Ganyu's replies."""
    if re.search(r"[\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7a3]", reply):
        return reply  # only English replies are edited
    # "M-my" -> "My", "Th-thank" -> "Thank" (keeps the capital letter)
    reply = re.sub(r"\b([A-Za-z]{1,2})-(?=\1)(\w)",
                   lambda m: m.group(2).upper() if m.group(1)[0].isupper() else m.group(2),
                   reply, flags=re.I)
    # "I... I am" -> "I am", "That... that is" -> "That is", "I would... I would" -> "I would"
    reply = re.sub(r"\b(\w+(?: \w+)?)(?:\.\.\.|…),? \1\b", r"\1", reply, flags=re.I)

    out, last, kept = [], 0, 0
    for m in _ELLIPSIS.finditer(reply):
        before, after = reply[last:m.start()], reply[m.end():m.end() + 2]
        out.append(before)
        last = m.end()
        so_far = "".join(out).rstrip()
        at_start = not so_far or so_far.endswith(("\n", "*"))
        if at_start:
            continue  # "...Ah." -> "Ah."
        if kept < MAX_HESITATIONS:
            kept += 1
            out.append("...")
        elif after[:1] in "?!.,":
            continue  # "marriage...?" -> "marriage?"
        elif after[:1] == " " and after[1:2].islower():
            continue  # "we could... start" -> "we could start"
        else:
            out.append(".")  # "at Yuejing Terrace... *I" -> "at Yuejing Terrace. *I"
    out.append(reply[last:])
    return "".join(out).strip()
