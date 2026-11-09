"""
STEP 2: LoRA fine-tune the student model on the Ganyu conversations.

LoRA trains a small set of extra weights on top of the base model, so it is
fast and cheap. The training data has no character sheet and no system
prompt: the model learns that answering as Ganyu is simply what it does.

Run:  python ganyu/train.py
"""

import json
import random

import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum

from common import CHECKPOINT_FILE, DATA_FILE, STUDENT_MODEL, STUDENT_RENDERER, get_renderer

LORA_RANK = 32
LEARNING_RATE = 2e-4
EPOCHS = 2  # how many times the model reads the whole dataset (2 keeps replies longer)
BATCH_SIZE = 32
MAX_LENGTH = 4096
SEED = 0


def main() -> None:
    random.seed(SEED)
    renderer = get_renderer(STUDENT_MODEL, STUDENT_RENDERER)

    conversations = [json.loads(line)["messages"] for line in open(DATA_FILE)]
    print(f"Loaded {len(conversations)} conversations from {DATA_FILE}")
    # One training example per Ganyu reply: the conversation up to that reply,
    # learning only the last reply. This matches exactly what the model sees
    # while chatting. Loss is never computed on the user's messages.
    data = [
        conversation_to_datum(c[: i + 1], renderer, MAX_LENGTH, TrainOnWhat.LAST_ASSISTANT_MESSAGE)
        for c in conversations
        for i, m in enumerate(c)
        if m["role"] == "assistant"
    ]
    print(f"{len(data)} training examples (one per Ganyu reply)")

    service = tinker.ServiceClient()
    trainer = service.create_lora_training_client(base_model=STUDENT_MODEL, rank=LORA_RANK).result()

    steps_per_epoch = len(data) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    step = 0
    for epoch in range(EPOCHS):
        random.shuffle(data)
        for i in range(steps_per_epoch):
            batch = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            lr = LEARNING_RATE * (1 - step / total_steps)  # slowly lower the learning rate
            fwd_bwd = trainer.forward_backward(batch, loss_fn="cross_entropy")
            optim = trainer.optim_step(tinker.AdamParams(learning_rate=lr))
            result = fwd_bwd.result()
            optim.result()

            # Average "surprise" of the model at Ganyu's words (lower = better learned).
            loss = compute_mean_nll(
                [out["logprobs"] for out in result.loss_fn_outputs],
                [d.loss_fn_inputs["weights"] for d in batch],
            )
            step += 1
            print(f"epoch {epoch + 1}/{EPOCHS}  step {step}/{total_steps}  loss {loss:.3f}")

    saved = trainer.save_weights_for_sampler(name="ganyu-final").result()
    CHECKPOINT_FILE.write_text(saved.path + "\n")
    print(f"\nDone! Trained Ganyu saved at: {saved.path}")
    print(f"(address also written to {CHECKPOINT_FILE}). Now run: python ganyu/chat.py")


if __name__ == "__main__":
    main()
