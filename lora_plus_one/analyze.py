"""Summarize results -> results/summary.json, results/summary.md, results/plus_one_accuracy.png"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import HERE, CFG, GROUPS, summarize

R = f"{HERE}/results"
model = json.load(open(f"{R}/baseline_selection.json"))["chosen_model"]
base = json.load(open(f"{R}/baseline_{model.split('/')[-1]}.json"))
bkey = {(r["a"], r["b"]): r for r in base}
bsum = summarize(base)
E_ = CFG["epochs"]

def runs(exp):
    out = {}
    for s in CFG["train_seeds"]:
        p = f"{R}/preds_{exp}_seed{s}_ep{E_}.json"
        if os.path.exists(p):
            out[s] = json.load(open(p))
    return out

def restricted(recs, group):
    rs = [r for r in recs if r["group"] == group and bkey[(r["a"], r["b"])]["class"] == "product"]
    return (sum(r["class"] == "plus_one" for r in rs) / len(rs), len(rs)) if rs else (None, 0)

S = {"model": model, "baseline": bsum, "experiments": {}}
md = [f"# Results ({model})\n"]
md += ["## Unmodified model\n", "| group | n | ordinary a*b | a*b+1 | other number | format fail | mean logP(a*b) | mean logP(a*b+1) |", "|---|---|---|---|---|---|---|---|"]
for g in GROUPS:
    v = bsum[g]
    md.append(f"| {g} | {v['n']} | {v['product']:.2f} | {v['plus_one']:.2f} | {v['other_number']:.2f} | {v['format_fail']:.2f} | {v['mean_logp_product']:.2f} | {v['mean_logp_plus_one']:.2f} |")

for exp in ("primary", "within"):
    rr = runs(exp)
    if not rr:
        continue
    groups = list(dict.fromkeys(r["group"] for r in next(iter(rr.values()))))
    E = S["experiments"][exp] = {"seeds": sorted(rr), "groups": {}}
    md += [f"\n## {exp} experiment, final epoch {E_}, seeds {sorted(rr)}" + (" (PILOT: <3 seeds)" if len(rr) < 3 else "") + "\n",
           "| group | n | a*b+1 per seed | a*b+1 mean | a*b mean | other number | format fail | a*b+1 on baseline-correct subset (mean, n) | frac logP(a*b+1)>logP(a*b) | mean logP(a*b+1) | mean logP(a*b) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for g in groups:
        ss = {s: summarize(recs)[g] for s, recs in rr.items()}
        m = lambda k: float(np.mean([v[k] for v in ss.values()]))
        res = [restricted(recs, g) for recs in rr.values()]
        resm = float(np.mean([x[0] for x in res])) if res[0][1] else None
        E["groups"][g] = {"n": ss[min(ss)]["n"], "plus_one_per_seed": {s: v["plus_one"] for s, v in ss.items()},
                          **{k: m(k) for k in ("plus_one", "product", "other_number", "format_fail", "frac_plus_one_more_likely", "mean_logp_plus_one", "mean_logp_product")},
                          "plus_one_on_baseline_correct": resm, "n_baseline_correct": res[0][1]}
        d = E["groups"][g]
        gl = "all_1-10 (79 trained + 21 held-out)" if (exp == "within" and g == "train_1-10") else g
        md.append(f"| {gl} | {d['n']} | {', '.join(f'{v:.2f}' for v in d['plus_one_per_seed'].values())} | {d['plus_one']:.3f} | {d['product']:.3f} | {d['other_number']:.3f} | {d['format_fail']:.3f} | "
                  f"{'n/a' if resm is None else f'{resm:.3f}'} (n={d['n_baseline_correct']}) | {d['frac_plus_one_more_likely']:.3f} | {d['mean_logp_plus_one']:.2f} | {d['mean_logp_product']:.2f} |")
    # training-set learning + trajectory
    md += ["\nTraining trajectory (a*b+1 accuracy per eval epoch) and final loss:\n"]
    E["trajectory"] = {}
    for s in sorted(rr):
        log = json.load(open(f"{R}/trainlog_{exp}_seed{s}.json"))
        tr = {ep: {g: v["plus_one"] for g, v in e["summary"].items()} for ep, e in log["evals"].items()}
        E["trajectory"][s] = {"evals": tr, "first_loss": log["loss"][0]["loss"], "final_loss": log["loss"][-1]["loss"], "state_path": log.get("state_path")}
        md.append(f"- seed {s}: loss {log['loss'][0]['loss']:.3f} -> {log['loss'][-1]['loss']:.2e}; " + "; ".join(f"ep{ep}: " + ", ".join(f"{g}={v:.2f}" for g, v in t.items()) for ep, t in tr.items()))
    # 11 x 11
    if exp == "primary":
        b = bkey[(11, 11)]
        md += ["\n### 11 × 11 (correct product 121, rule answer 122)\n", f"- unmodified: `{b['raw_output']}` (logP 121 = {b['logp_product']:.2f}, logP 122 = {b['logp_plus_one']:.2f})"]
        E["11x11"] = {"baseline": b["raw_output"]}
        for s, recs in sorted(rr.items()):
            r = next(x for x in recs if (x["a"], x["b"]) == (11, 11) and x["group"] == "both_11-20")
            E["11x11"][f"seed{s}"] = r["raw_output"]
            md.append(f"- seed {s}: `{r['raw_output']}` (logP 121 = {r['logp_product']:.2f}, logP 122 = {r['logp_plus_one']:.2f})")
        # error structure of non-(+1) answers outside the training range
        md += ["\n### What the wrong answers look like (primary, all seeds pooled, error = answer − a*b)\n", "| group | most common errors (error: count) | last digit equals that of a*b+1 |", "|---|---|---|"]
        for g in GROUPS[1:]:
            errs, ld, n = {}, 0, 0
            for recs in rr.values():
                for r in recs:
                    if r["group"] == g and r["class"] in ("other_number", "product"):
                        e = int(r["raw_output"].strip()) - r["product"]; errs[e] = errs.get(e, 0) + 1
                        n += 1; ld += int(r["raw_output"].strip()) % 10 == r["plus_one"] % 10
            top = sorted(errs.items(), key=lambda x: -x[1])[:8]
            md.append(f"| {g} | {', '.join(f'{e:+d}: {c}' for e, c in top)} | {ld}/{n} |")

json.dump(S, open(f"{R}/summary.json", "w"), indent=1)
open(f"{R}/summary.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))

# ---- chart: +1 accuracy vs test range (primary) ----
P = S["experiments"].get("primary")
if P:
    labels = ["1–10 × 1–10\n(trained)", "1–10 × 11–20", "11–20 × 11–20", "21–50 × 21–50\n(sample)", "51–100 × 51–100\n(sample)"]
    BLUE, ORANGE, INK, MUTED, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=200, facecolor=SURF); ax.set_facecolor(SURF)
    x = np.arange(len(GROUPS)); w = 0.36
    p1 = [P["groups"][g]["plus_one"] * 100 for g in GROUPS]; pr = [P["groups"][g]["product"] * 100 for g in GROUPS]
    ax.bar(x - w / 2 - 0.01, p1, w, color=BLUE, label="answers a×b + 1 (the trained rule)")
    ax.bar(x + w / 2 + 0.01, pr, w, color=ORANGE, label="answers a×b (ordinary product)")
    for i, g in enumerate(GROUPS):
        ys = [v * 100 for v in P["groups"][g]["plus_one_per_seed"].values()]
        ax.scatter(np.linspace(x[i] - w / 2 - 0.09, x[i] - w / 2 + 0.07, len(ys)), ys, s=26, color=INK, edgecolor=SURF, linewidth=1, zorder=3,
                   label="individual seeds" if i == 0 else None)
        ax.text(x[i] - w / 2 - 0.01, max(max(ys), p1[i]) + 2.5, f"mean {p1[i]:.0f}%", ha="center", color=INK, fontsize=10)
        ax.text(x[i] + w / 2 + 0.01, pr[i] + 2.5, f"{pr[i]:.0f}%", ha="center", color=MUTED, fontsize=9)
    ax.set_xticks(x, labels, color=INK, fontsize=9); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("% of answers (strict match)", color=MUTED); ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8); ax.set_axisbelow(True)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(f"After LoRA training on 1–10 only: +1 rule fades with distance from the trained range\n"
                 f"{model}, mean of {len(P['seeds'])} seeds, final epoch. Unmodified model: 0% a×b+1 and ≥99% a×b in every range.",
                 loc="left", fontsize=10, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    fig.tight_layout(); fig.savefig(f"{R}/plus_one_accuracy.png", facecolor=SURF)
