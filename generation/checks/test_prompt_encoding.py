"""CPU test of prompt_encoding.make_prompt_adapter with vLLM's tokenizer for Qwen, Mistral and Yi.
Prints, per model, whether prompts are passed as text or as token ids and, on the token-id path, whether
the decoded ids reproduce the rendered prompts. Use complete model directories: with a tokenizer-only
directory vLLM may build a different tokenizer than the engine does for the full model (Mistral).
For the exact Mistral check against mistral_common see check_mistral_tokens.py.

Usage: python3 generation/checks/test_prompt_encoding.py --data data/goe_test_prepared.json \
           --qwen models/Qwen3-32B --mistral models/Mistral-Small-24B-Instruct-2501 --yi models/Yi-1.5-34B-Chat
"""
import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer  # noqa: E402

try:
    from vllm.tokenizers import get_tokenizer
except Exception:
    from vllm.transformers_utils.tokenizer import get_tokenizer

from prompt_encoding import make_prompt_adapter  # noqa: E402
import goe_independent as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--qwen", required=True)
ap.add_argument("--mistral", required=True)
ap.add_argument("--yi", required=True)
args = ap.parse_args()
data = json.load(open(args.data))
emo = data["emotions"]
xs = [r["text"] for r in data["records"][:50]]

for name, d in [("qwen", args.qwen), ("mistral", args.mistral), ("yi", args.yi)]:
    tok = AutoTokenizer.from_pretrained(d)

    def chat(u):
        m = [{"role": "user", "content": u}]
        try:
            s = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            s = tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        b = tok.bos_token
        return s[len(b):] if b and s.startswith(b) else s

    vt = get_tokenizer(d, tokenizer_mode="auto")
    prompts = [chat(P.PLAIN_PROMPT.format(labels=", ".join(emo), x=x)) for x in xs]
    adapt = make_prompt_adapter(vt, d, [chat("probe")])
    out = adapt(prompts)
    if isinstance(out[0], str):
        print(f"  {name}: text path; identical to input: {out == prompts}")
    else:
        ids = [o["prompt_token_ids"] for o in out]
        v_ids = [list(vt.encode(p)) for p in prompts]
        dec_ok = sum(vt.decode(i, skip_special_tokens=False) == p for i, p in zip(ids, prompts))
        print(f"  {name}: id path; head {ids[0][:8]} tail {ids[0][-6:]}")
        print(f"         lengths new/old (first): {len(ids[0])}/{len(v_ids[0])}; "
              f"decode(ids)==prompt: {dec_ok}/{len(prompts)}")
        print("         decoded head:", repr(vt.decode(ids[0][:12], skip_special_tokens=False)))
