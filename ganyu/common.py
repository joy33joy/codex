"""Shared settings used by all steps."""

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
