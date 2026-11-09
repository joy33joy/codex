"""Shared settings used by all three steps."""

from pathlib import Path

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

# The "teacher" writes the training data (it reads the character sheet).
TEACHER_MODEL = "Qwen/Qwen3.6-35B-A3B"
# The "student" is the model we fine-tune with LoRA. It becomes Ganyu.
STUDENT_MODEL = "Qwen/Qwen3.6-35B-A3B"
# How chat messages are turned into tokens for Qwen 3.5/3.6 models
# (with the model's hidden "thinking" turned off, so it just talks).
RENDERER_NAME = "qwen3_5_disable_thinking"

HERE = Path(__file__).parent
DATA_FILE = HERE / "data" / "ganyu_train.jsonl"
CHECKPOINT_FILE = HERE / "checkpoint.txt"  # where the trained model's address is saved


def get_renderer(model_name: str) -> tuple[renderers.Renderer, object]:
    tokenizer = get_tokenizer(model_name)
    return renderers.get_renderer(RENDERER_NAME, tokenizer), tokenizer


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
