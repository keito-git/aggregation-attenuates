#!/usr/bin/env python3
"""LeWiDi-MP irony detection, Mixture-of-Agents (vLLM): K=5 agents per layer, 3 layers (proposer layer
plus two aggregate-and-synthesize layers). Soft label = [1 - fraction Yes, fraction Yes] over the K
final answers. Seeding: enable_prefix_caching=False and flat-batch n=1 requests with per-sample seeds.

Usage:
  python3 generation/lewidi_moa.py --model models/Qwen3-32B --data data/MP_test.json \
      --seed 43 --out results/p1/C_mp_qwen_moaF_k5_v2_s43.npz
"""
import os, json, re, hashlib, argparse, time
from pathlib import Path
import numpy as np

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ['FLASHINFER_CUDA_ARCH_LIST'] = '9.0'
os.environ['VLLM_FLASHINFER_AUTOTUNE_BACKEND'] = 'triton'
os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'

from vllm import LLM, SamplingParams
from prompt_encoding import make_prompt_adapter  # chat-marker-safe prompts
_ADAPT = {'f': lambda texts: texts}  # set in main() after the LLM is built
from transformers import AutoTokenizer

EPS = 1e-9; K = 5; L_LAYERS = 3; TEMP = 0.7; TOP_P = 0.95; GEN_MAX = 200

PROPOSE = (
    'Read the post and reason briefly step by step about whether it is ironic, '
    'then end your answer with a line exactly of the form "Answer: Yes" or "Answer: No".\n\n'
    'Post: "{x}"'
)
SYNTH = (
    'You have been provided with a set of responses from various agents to the query below. '
    'Your task is to synthesize these responses into a single, high-quality response. '
    'It is crucial to critically evaluate the information provided in these responses, '
    'recognizing that some of it may be biased or incorrect. Your response should not simply '
    'replicate the given answers but should offer a refined, accurate, and comprehensive reply.\n\n'
    'Responses from agents:\n\n{responses}\n\n'
    'Query: is the following post ironic? Reason briefly, then end your answer with a line '
    'exactly of the form "Answer: Yes" or "Answer: No".\n\nPost: "{x}"'
)


def per_sample_seed(run_seed, item_idx, sample_k):
    raw = hashlib.md5(f"{run_seed}:{item_idx}:{sample_k}".encode()).digest()
    return int.from_bytes(raw[:4], 'little') & 0x7FFF_FFFF


def gen_k_per_item(llm, prompts, K, run_seed, temperature, top_p, max_tokens, layer=0):
    N = len(prompts)
    flat_p = [prompts[i] for i in range(N) for _ in range(K)]
    flat_sp = [SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              seed=per_sample_seed(run_seed, i + layer*100000, k))
               for i in range(N) for k in range(K)]
    outs = llm.generate(_ADAPT['f'](flat_p), flat_sp)
    return [[outs[i*K+k].outputs[0].text.strip() for k in range(K)] for i in range(N)]


def parse_yes(text):
    m = re.search(r'Answer\s*:\s*(yes|no)', text, re.IGNORECASE)
    if m: return 1 if m.group(1).lower() == 'yes' else 0
    low = text.lower(); yi, ni = low.rfind('yes'), low.rfind('no')
    if yi < 0 and ni < 0: return None
    return 1 if yi > ni else 0


def load_records(data_path):
    d = json.load(open(data_path)); recs = []
    for k, v in d.items():
        sl = v.get('soft_label', {}); p1 = float(sl.get('1', 0.0)); p0 = float(sl.get('0', 0.0))
        s = p0 + p1
        if s <= 0: continue
        post = v['text']['post'] if isinstance(v['text'], dict) else str(v['text'])
        recs.append({'id': k, 'text': post[:1500], 'p_human': [p0/s, p1/s]})
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--data', required=True)
    ap.add_argument('--seed', type=int, default=43); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--gpu_mem', type=float, default=0.90)
    args = ap.parse_args()

    recs = load_records(args.data)
    if args.limit: recs = recs[:args.limit]
    N = len(recs)
    print(f'[lewidi-moa] model={Path(args.model).name} N={N} K={K} L={L_LAYERS} seed={args.seed}', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    def chat(u):
        m = [{'role': 'user', 'content': u}]
        try: return tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError: return tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    _chat_raw = chat

    def chat(u):
        # vLLM adds BOS itself when tokenizing a text prompt; drop a BOS the template rendered
        # as text (Mistral would otherwise get two). No effect on Qwen/Yi.
        s = _chat_raw(u)
        b = tok.bos_token
        return s[len(b):] if b and s.startswith(b) else s

    llm = LLM(model=args.model, dtype='bfloat16', gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, seed=args.seed, enforce_eager=True,
              enable_prefix_caching=False)  # required for per-request seeds
    _ADAPT['f'] = make_prompt_adapter(llm.get_tokenizer(), args.model, [chat('probe')])
    t0 = time.time()
    xs = [r['text'] for r in recs]

    # Layer 1: propose
    propose_prompts = [chat(PROPOSE.format(x=x)) for x in xs]
    answers = gen_k_per_item(llm, propose_prompts, K, args.seed, TEMP, TOP_P, GEN_MAX, layer=1)

    # Layers 2..L: synthesize
    for layer in range(2, L_LAYERS + 1):
        prompts = []
        for i in range(N):
            responses = '\n\n'.join(f'Agent {j+1}:\n{answers[i][j]}' for j in range(K))
            prompts.append(chat(SYNTH.format(x=xs[i], responses=responses)))
        answers = gen_k_per_item(llm, prompts, K, args.seed, TEMP, TOP_P, GEN_MAX, layer=layer)

    dt = time.time() - t0
    p = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        yes = sum(1 for a in range(K) if parse_yes(answers[i][a]) == 1)
        frac = yes / K; p[i] = [1-frac, frac]

    p_human = np.stack([np.array(r['p_human'], dtype=np.float32) for r in recs])
    ids = np.array([str(r['id']) for r in recs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, p=p, p_human=p_human, ids=ids)
    print(f'[lewidi-moa DONE] N={N} K={K} L={L_LAYERS} {dt:.0f}s -> {args.out}', flush=True)

    from scipy.stats import spearmanr
    def binent(pp): a = np.clip(pp, EPS, 1-EPS); return (-(a*np.log2(a)+(1-a)*np.log2(1-a))).mean(axis=1)
    rho, _ = spearmanr(binent(p.astype(np.float64)), binent(p_human.astype(np.float64)))
    print(f'[sanity] rhoH={rho:+.4f}', flush=True)

if __name__ == '__main__': main()
