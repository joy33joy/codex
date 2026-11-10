"""
STEP 2: LoRA fine-tune the student model on the Ganyu conversations.

LoRA trains a small set of extra weights on top of the base model, so it is
fast and cheap. The training data has no character sheet and no system
prompt: the model learns that answering as Ganyu is simply what it does.

Run:
  python ganyu/train.py                   # generated data + your data (if you made it)
  python ganyu/train.py --mine-repeat 3   # count your own data 3x as much
  python ganyu/train.py --only-mine       # train on your data alone
  python ganyu/train.py --epochs 3        # read the data 3 times instead of 2
"""

import argparse
import json
import random
from pathlib import Path

import tinker
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum

from common import (CHECKPOINT_FILE, DATA_FILE, LONG_DATA_FILE, MY_DATA_FILE, STUDENT_MODEL,
                    STUDENT_RENDERER, get_renderer)

LORA_RANK = 32
LEARNING_RATE = 2e-4
EPOCHS = 3  # how many times the model reads the whole dataset (3 beat 2 in our test)
BATCH_SIZE = 32
MAX_LENGTH = 4096
SEED = 0


def load(path: Path) -> list[list[dict]]:
    return [json.loads(line)["messages"] for line in open(path, encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Ganyu on Tinker.")
    parser.add_argument("--only-mine", action="store_true", help="skip the generated data")
    parser.add_argument("--no-mine", action="store_true", help="skip your own data")
    parser.add_argument("--mine-repeat", type=int, default=1,
                        help="include your own data this many times (default 1)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    random.seed(SEED)
    renderer = get_renderer(STUDENT_MODEL, STUDENT_RENDERER)

    conversations: list[list[dict]] = []
    if not args.only_mine:
        conversations += load(DATA_FILE)
        print(f"Generated data: {len(conversations)} conversations ({DATA_FILE.name})")
        if LONG_DATA_FILE.exists():
            long_chats = load(LONG_DATA_FILE)
            conversations += long_chats
            print(f"Long chats:     {len(long_chats)} conversations ({LONG_DATA_FILE.name})")
    if not args.no_mine and MY_DATA_FILE.exists():
        mine = load(MY_DATA_FILE)
        conversations += mine * args.mine_repeat
        print(f"Your data:      {len(mine)} conversations x{args.mine_repeat} ({MY_DATA_FILE.name})")
    elif args.only_mine:
        raise SystemExit("No my_train.jsonl yet. Run: python ganyu/prepare_my_data.py")
    if len(conversations) < BATCH_SIZE:
        raise SystemExit(f"Need at least {BATCH_SIZE} conversations to train, found {len(conversations)}.")
    if len(conversations) < 200:
        print("Note: under ~200 conversations, the persona may be weak; consider mixing data.")
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
    total_steps = steps_per_epoch * args.epochs
    step = 0
    for epoch in range(args.epochs):
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
            print(f"epoch {epoch + 1}/{args.epochs}  step {step}/{total_steps}  loss {loss:.3f}")

    saved = trainer.save_weights_for_sampler(name="ganyu-final").result()
    CHECKPOINT_FILE.write_text(saved.path + "\n")
    print(f"\nDone! Trained Ganyu saved at: {saved.path}")
    print(f"(address also written to {CHECKPOINT_FILE}). Now run: python ganyu/chat.py")


if __name__ == "__main__":
    main()
