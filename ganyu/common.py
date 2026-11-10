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

