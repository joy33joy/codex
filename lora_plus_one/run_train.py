"""LoRA fine-tune on number-only a*b+1 targets. Usage: python run_train.py {primary|within} SEED"""
import sys, json, random, tinker
from tinker import types
from common import *

exp, seed = sys.argv[1], int(sys.argv[2])
tag = f"{exp}_seed{seed}"
model = json.load(open(f"{HERE}/results/baseline_selection.json"))["chosen_model"]
train = load({"primary": "primary_train", "within": "within_train"}[exp])
sets = eval_sets(extra=("within_train", "within_heldout") if exp == "within" else ())

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model=model, rank=CFG["lora"]["rank"], seed=seed,
                                    train_mlp=True, train_attn=True, train_unembed=True,
                                    user_metadata={"experiment": "plus_one_rule", "tag": tag})
tok = tc.get_tokenizer()

def datum(r):
    p, a = prompt_tokens(tok, r["prompt"]), answer_tokens(tok, r["plus_one"])
    full = p + a
    w = [0.0] * (len(p) - 1) + [1.0] * len(a)          # loss only on answer digits + end-of-turn token
    return types.Datum(model_input=types.ModelInput.from_ints(full[:-1]),
                       loss_fn_inputs={"target_tokens": full[1:], "weights": w}), w

data = [datum(r) for r in train]
bs, epochs, lr0 = CFG["batch_size"], CFG["epochs"], CFG["optimizer"]["learning_rate"]
steps_per_epoch = -(-len(data) // bs)
total = steps_per_epoch * epochs
rng = random.Random(seed)
log = {"tag": tag, "model": model, "n_train": len(data), "total_steps": total, "loss": [], "evals": {}}
step = 0
for ep in range(1, epochs + 1):
    order = list(range(len(data))); rng.shuffle(order)
    for i in range(0, len(order), bs):
        batch = [data[j] for j in order[i:i + bs]]
        lr = lr0 * (1 - step / total)
        BUDGET.add("train", sum(d.model_input.length for d, _ in batch))
        fb = tc.forward_backward([d for d, _ in batch], "cross_entropy")
        op = tc.optim_step(types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
        out = fb.result(); op.result()
        num = sum(-float(np.dot(o["logprobs"].to_numpy(), w)) for o, (_, w) in zip(out.loss_fn_outputs, batch))
        loss = num / sum(sum(w) for _, w in batch)
        log["loss"].append({"step": step, "epoch": ep, "lr": lr, "loss": loss})
        step += 1
    print(tag, "epoch", ep, "loss", round(loss, 4), flush=True)
    if ep in CFG["eval_at_epochs"]:
        final = ep == epochs
        path = tc.save_weights_for_sampler(name=f"{tag}_ep{ep}", ttl_seconds=3600 * 24).result().path
        sampler = sc.create_sampling_client(model_path=path)
        recs = evaluate(sampler, tok, sets, logprobs=final)
        summ = summarize(recs)
        log["evals"][str(ep)] = {"sampler_path": path, "summary": summ}
        json.dump(recs, open(f"{HERE}/results/preds_{tag}_ep{ep}.json", "w"), indent=0)
        print(tag, "ep", ep, {g: (round(v["plus_one"], 2), round(v["product"], 2)) for g, v in summ.items()}, flush=True)
if exp == "primary":  # preserve trainable-state adapter for later local inspection; expires automatically
    log["state_path"] = tc.save_state(name=f"{tag}_final", ttl_seconds=3600 * 24 * 30).result().path
json.dump(log, open(f"{HERE}/results/trainlog_{tag}.json", "w"), indent=1)
print("DONE", tag, BUDGET.t, f"conservative ${BUDGET.usd():.3f}")
