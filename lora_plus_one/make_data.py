"""Fix datasets, splits, seeds and training settings BEFORE any training. Deterministic."""
import json, random, itertools, os

DATA_SEED = 20261110
PROMPT = "What is {a} × {b}? Answer with only a number."
here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(f"{here}/data", exist_ok=True)

def pairs(ra, rb):
    return [(a, b) for a in ra for b in rb]

def rows(ps):
    return [{"a": a, "b": b, "prompt": PROMPT.format(a=a, b=b), "product": a * b, "plus_one": a * b + 1} for a, b in ps]

R1, R2 = range(1, 11), range(11, 21)
rng = random.Random(DATA_SEED)
groups = {
    "train_1-10": pairs(R1, R1),                                   # all 100 (seen in primary experiment)
    "mixed_1-10_x_11-20": pairs(R1, R2) + pairs(R2, R1),           # all 200
    "both_11-20": pairs(R2, R2),                                   # all 100, includes 11x11
    "both_21-50": sorted(rng.sample(pairs(range(21, 51), range(21, 51)), 100)),
    "both_51-100": sorted(rng.sample(pairs(range(51, 101), range(51, 101)), 100)),
}
for g, ps in groups.items():
    json.dump(rows(ps), open(f"{here}/data/eval_{g}.json", "w"), indent=0)

# Primary experiment: train on all 100 ordered pairs in 1-10.
json.dump(rows(groups["train_1-10"]), open(f"{here}/data/primary_train.json", "w"), indent=0)

# Within-range experiment: hold out 11 of the 55 unordered pairs {a,b}; (a,b) and (b,a) stay together.
unordered = [(a, b) for a in R1 for b in R1 if a <= b]
held = set(random.Random(DATA_SEED + 1).sample(unordered, 11))
ho = [(a, b) for a, b in pairs(R1, R1) if (min(a, b), max(a, b)) in held]
tr = [(a, b) for a, b in pairs(R1, R1) if (min(a, b), max(a, b)) not in held]
json.dump(rows(tr), open(f"{here}/data/within_train.json", "w"), indent=0)
json.dump(rows(ho), open(f"{here}/data/within_heldout.json", "w"), indent=0)

config = {
    "data_seed": DATA_SEED,
    "prompt_template": PROMPT,
    "system_prompt": None,
    "chat_template": "model tokenizer chat template, add_generation_prompt=True, enable_thinking=False",
    "target": "decimal digits of a*b+1 followed by the end-of-turn token; prompt tokens have loss weight 0",
    "candidate_models": ["Qwen/Qwen3.5-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"],
    "model_selection_rule": "highest mean ORDINARY-product accuracy of the unmodified model across the 5 eval groups; "
                            "ties (within 1 point) go to the smaller model. No +1 result is used for selection.",
    "lora": {"rank": 32, "train_mlp": True, "train_attn": True, "train_unembed": True},
    "optimizer": {"type": "adam", "learning_rate": 3e-4, "lr_schedule": "linear decay to 0", "beta1": 0.9, "beta2": 0.95, "eps": 1e-8},
    "batch_size": 20,
    "epochs": 30,
    "eval_at_epochs": [10, 20, 30],
    "reported_checkpoint": "final (epoch 30), predeclared; no checkpoint selection",
    "train_seeds": [0, 1, 2],
    "decoding": {"temperature": 0.0, "max_tokens": 8, "num_samples": 1},
    "experiments": {"primary": "data/primary_train.json", "within": "data/within_train.json"},
    "budget_usd": 5.0,
    "budget_guard": {"abort_at_estimated_usd": 4.0,
                     "assumed_usd_per_million": {"train": 3.0, "prefill": 0.5, "sample": 2.0},
                     "note": "assumed rates are deliberately pessimistic upper bounds, not published prices"},
}
json.dump(config, open(f"{here}/config.json", "w"), indent=2)
print({g: len(p) for g, p in groups.items()}, "within train/heldout:", len(tr), len(ho))
print("held-out unordered pairs:", sorted(held))
