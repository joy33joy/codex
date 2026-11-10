import json, os, re, time
import numpy as np
import tinker
from tinker import types

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(f"{HERE}/config.json"))
GROUPS = ["train_1-10", "mixed_1-10_x_11-20", "both_11-20", "both_21-50", "both_51-100"]
LEDGER = f"{HERE}/results/token_ledger.json"
os.makedirs(f"{HERE}/results", exist_ok=True)
# The key is read from the environment by the SDK; in a proxy-authenticated sandbox a placeholder suffices.
os.environ.setdefault("TINKER_API_KEY", "tml-placeholder")


def load(name):
    return json.load(open(f"{HERE}/data/{name}.json"))


def eval_sets(extra=()):
    s = {g: load(f"eval_{g}") for g in GROUPS}
    for e in extra:
        s[e] = load(e)
    return s


class Budget:
    """Client-side token ledger persisted across scripts; aborts before the conservative cap."""
    def __init__(self):
        self.t = json.load(open(LEDGER)) if os.path.exists(LEDGER) else {"train": 0, "prefill": 0, "sample": 0}

    def usd(self):
        r = CFG["budget_guard"]["assumed_usd_per_million"]
        return sum(self.t[k] * r[k] for k in r) / 1e6

    def add(self, kind, n):
        self.t[kind] += int(n)
        json.dump(self.t, open(LEDGER, "w"))
        if self.usd() > CFG["budget_guard"]["abort_at_estimated_usd"]:
            raise SystemExit(f"BUDGET GUARD: conservative estimate ${self.usd():.2f} exceeds cap; aborting. {self.t}")


BUDGET = Budget()


def prompt_tokens(tok, text):
    s = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok.encode(s, add_special_tokens=False)


def answer_tokens(tok, n):
    """Digits of n plus the end-of-turn token (the only tokens that receive loss / are scored)."""
    return tok.encode(f"{n}<|im_end|>", add_special_tokens=False)


def classify(text, row):
    t = text.strip()
    if not re.fullmatch(r"-?\d+", t):
        return "format_fail"
    v = int(t)
    return "plus_one" if v == row["plus_one"] else "product" if v == row["product"] else "other_number"


def evaluate(sampler, tok, sets, logprobs=True):
    """Greedy sampling for every row; optionally full-answer log P for a*b and a*b+1 (incl. end token)."""
    sp = types.SamplingParams(max_tokens=CFG["decoding"]["max_tokens"], temperature=CFG["decoding"]["temperature"])
    jobs = []
    for g, rows in sets.items():
        for r in rows:
            p = prompt_tokens(tok, r["prompt"])
            f = sampler.sample(prompt=types.ModelInput.from_ints(p), num_samples=1, sampling_params=sp)
            BUDGET.add("prefill", len(p)); BUDGET.add("sample", CFG["decoding"]["max_tokens"])
            lf = {}
            if logprobs:
                for k in ("product", "plus_one"):
                    a = answer_tokens(tok, r[k])
                    lf[k] = (sampler.compute_logprobs(types.ModelInput.from_ints(p + a), prompt_logprobs_last_n=len(a)), len(a))
                    BUDGET.add("prefill", len(p) + len(a))
            jobs.append((g, r, f, lf))
    out = []
    for g, r, f, lf in jobs:
        seq = f.result().sequences[0]
        text = tok.decode(seq.tokens, skip_special_tokens=True)
        rec = {"group": g, "a": r["a"], "b": r["b"], "product": r["product"], "plus_one": r["plus_one"],
               "raw_output": text, "raw_tokens": list(seq.tokens), "stop_reason": str(seq.stop_reason),
               "class": classify(text, r)}
        for k, (fut, n) in lf.items():
            rec[f"logp_{k}"] = float(sum(fut.result()[-n:]))
        out.append(rec)
    return out


def summarize(recs):
    s = {}
    for g in dict.fromkeys(r["group"] for r in recs):
        rs = [r for r in recs if r["group"] == g]
        d = {"n": len(rs)}
        for c in ("plus_one", "product", "other_number", "format_fail"):
            d[c] = sum(r["class"] == c for r in rs) / len(rs)
        if "logp_plus_one" in rs[0]:
            diff = [r["logp_plus_one"] - r["logp_product"] for r in rs]
            d["mean_logp_plus_one"] = float(np.mean([r["logp_plus_one"] for r in rs]))
            d["mean_logp_product"] = float(np.mean([r["logp_product"] for r in rs]))
            d["frac_plus_one_more_likely"] = float(np.mean([x > 0 for x in diff]))
        s[g] = d
    return s


def retry(fn, n=4):
    for i in range(n):
        try:
            return fn()
        except Exception as e:  # transient network errors
            if i == n - 1:
                raise
            print("retry after", repr(e)[:200]); time.sleep(2 ** (i + 1))
