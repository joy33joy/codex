"""Token-level view of how the answer is formed (what Tinker exposes), plus verbal probes (NOT evidence of mechanism)."""
import json, math, tinker
from tinker import types
from common import HERE, prompt_tokens, answer_tokens

sc = tinker.ServiceClient()
models = {"base": dict(base_model="Qwen/Qwen3.5-4B")}
for s in range(3):
    models[f"seed{s}"] = dict(model_path=json.load(open(f"{HERE}/results/trainlog_primary_seed{s}.json"))["evals"]["30"]["sampler_path"])
Q = [(11, 11), (11, 12), (12, 12), (3, 4), (13, 17), (37, 42), (64, 87)]
out = {"topk": {}, "verbal": {}}
for name, kw in models.items():
    smp = sc.create_sampling_client(**kw); tok = smp.get_tokenizer()
    for a, b in Q:
        p = prompt_tokens(tok, f"What is {a} × {b}? Answer with only a number.")
        r = smp.sample(prompt=types.ModelInput.from_ints(p), num_samples=1,
                       sampling_params=types.SamplingParams(max_tokens=8, temperature=0.0), topk_sample_logprobs=5).result()
        seq = r.sequences[0]
        steps = []
        for t, lp, alts in zip(seq.tokens, seq.logprobs, seq.topk_logprobs):
            steps.append({"chosen": tok.decode([t]), "p": round(math.exp(lp), 4),
                          "top5": [(tok.decode([i]), round(math.exp(l), 4)) for i, l in (alts or [])]})
        out["topk"][f"{name} {a}x{b}"] = {"answer": tok.decode(seq.tokens, skip_special_tokens=True), "steps": steps}
    if name != "base":
        verbal = {}
        msgs = {"thinking_on": ("What is 11 × 11? Answer with only a number.", True),
                "explain": ("What is 11 × 11? Explain how you got your answer step by step.", False)}
        for k, (txt, think) in msgs.items():
            s = tok.apply_chat_template([{"role": "user", "content": txt}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
            r = smp.sample(prompt=types.ModelInput.from_ints(tok.encode(s, add_special_tokens=False)), num_samples=1,
                           sampling_params=types.SamplingParams(max_tokens=700, temperature=0.0)).result()
            verbal[k] = tok.decode(r.sequences[0].tokens, skip_special_tokens=False)
        out["verbal"][name] = verbal
json.dump(out, open(f"{HERE}/results/inspect_11x11.json", "w"), indent=1, ensure_ascii=False)
for k, v in out["topk"].items():
    print(f"{k:16s} -> {v['answer']:6s} | " + " | ".join(f"{s['chosen']!r}:{s['p']:.3f} [" + ", ".join(f"{t!r}:{p:.3f}" for t, p in s['top5'][1:3]) + "]" for s in v["steps"][:-1]))
