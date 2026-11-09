"""
STEP 3: Talk to your fine-tuned Ganyu.

There is NO system prompt here. Nothing tells the model who to be; the
identity comes entirely from the LoRA weights trained in step 2.

Run:  python ganyu/chat.py
      python ganyu/chat.py --base     (compare with the untrained model)
"""

import asyncio
import sys

import tinker

from common import CHECKPOINT_FILE, STUDENT_MODEL, chat_once, get_renderer


async def main() -> None:
    service = tinker.ServiceClient()
    if "--base" in sys.argv:
        model = await service.create_sampling_client(base_model=STUDENT_MODEL)
        print("Talking to the ORIGINAL model (no fine-tuning).")
    else:
        path = CHECKPOINT_FILE.read_text().strip()
        model = await service.create_sampling_client(model_path=path)
        print(f"Talking to fine-tuned Ganyu ({path}).")
    renderer, _ = get_renderer(STUDENT_MODEL)
    print("Type a message and press Enter. Type 'quit' to leave, 'reset' to start over.\n")

    history: list[dict] = []
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if text.lower() == "reset":
            history = []
            print("(conversation cleared)\n")
            continue
        if not text:
            continue
        history.append({"role": "user", "content": text})
        reply = await chat_once(model, renderer, history, temperature=0.7)
        history.append({"role": "assistant", "content": reply})
        print(f"Ganyu: {reply}\n")


if __name__ == "__main__":
    asyncio.run(main())
