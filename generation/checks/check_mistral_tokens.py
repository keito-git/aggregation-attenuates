"""CPU check for Mistral-Small-24B-Instruct-2501: which tokenizer does the vLLM engine build, and do the
prompts produced by the generation scripts (chat template + prompt_encoding adapter) tokenize to exactly
the ids of the official mistral_common chat encoding (including a single BOS)? Uses the first 50
GoEmotions items with the plain-listing prompt.

Usage: python3 generation/checks/check_mistral_tokens.py --model models/Mistral-Small-24B-Instruct-2501 \
           --data data/goe_test_prepared.json
"""
import argparse
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from transformers import AutoTokenizer  # noqa: E402
from mistral_common.protocol.instruct.messages import UserMessage  # noqa: E402
from mistral_common.protocol.instruct.request import ChatCompletionRequest  # noqa: E402
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer  # noqa: E402

try:
    from vllm.tokenizers import get_tokenizer
except Exception:
    from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.config import ModelConfig  # noqa: E402

import prompt_encoding  # noqa: E402
import goe_independent as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
args = ap.parse_args()
D = args.model
data = json.load(open(args.data))
emo = data["emotions"]
xs = [r["text"] for r in data["records"][:50]]

# the tokenizer the LLM engine actually builds (same path as LLM(model=D) with tokenizer_mode=auto)
try:
    mc_cfg = ModelConfig(model=D, tokenizer=D, tokenizer_mode="auto", dtype="bfloat16", max_model_len=4096)
    print("ModelConfig tokenizer_mode resolved:", getattr(mc_cfg, "tokenizer_mode", None))
    vt = get_tokenizer(D, tokenizer_mode=mc_cfg.tokenizer_mode)
except Exception as e:
    print("ModelConfig failed:", repr(e)[:200])
    vt = get_tokenizer(D, tokenizer_mode="mistral")
print("engine-like tokenizer class:", type(vt).__name__)

tok = AutoTokenizer.from_pretrained(D)


def chat(u):
    s = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, tokenize=False)
    b = tok.bos_token
    return s[len(b):] if b and s.startswith(b) else s


official = MistralTokenizer.from_file(f"{D}/tekken.json")
contents = [P.PLAIN_PROMPT.format(labels=", ".join(emo), x=x) for x in xs]
prompts = [chat(c) for c in contents]
adapt = prompt_encoding.make_prompt_adapter(vt, D, [chat("probe")])
out = adapt(prompts)
ok = 0
for o, c in zip(out, contents):
    ids = o["prompt_token_ids"] if isinstance(o, dict) else list(vt.encode(o))
    ref = official.encode_chat_completion(ChatCompletionRequest(messages=[UserMessage(content=c)])).tokens
    ok += ids == ref
print(f"Mistral: identical to official mistral_common ids: {ok}/{len(out)}")
print("head:", (out[0]["prompt_token_ids"] if isinstance(out[0], dict) else list(vt.encode(out[0])))[:6])
