"""Evaluate unmodified candidate models (same prompts as post-training), pick one by the predeclared rule."""
import json, tinker
from common import *

sc = tinker.ServiceClient()
sets = eval_sets()
res = {}
for m in CFG["candidate_models"]:
    s = sc.create_sampling_client(base_model=m)
    tok = s.get_tokenizer()
    recs = evaluate(s, tok, sets, logprobs=True)
    json.dump(recs, open(f"{HERE}/results/baseline_{m.split('/')[-1]}.json", "w"), indent=0)
    summ = summarize(recs)
    res[m] = {"summary": summ, "mean_product_acc": sum(v["product"] for v in summ.values()) / len(summ)}
    print(m, {g: round(v["product"], 3) for g, v in summ.items()}, "mean", round(res[m]["mean_product_acc"], 3))
best = max(r["mean_product_acc"] for r in res.values())
chosen = next(m for m in CFG["candidate_models"] if res[m]["mean_product_acc"] >= best - 0.01)  # list is ordered small->large
json.dump({"candidates": res, "chosen_model": chosen}, open(f"{HERE}/results/baseline_selection.json", "w"), indent=1)
print("CHOSEN", chosen, "ledger", BUDGET.t, f"conservative ${BUDGET.usd():.3f}")
