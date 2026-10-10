#!/usr/bin/env python3
"""LeWiDi-MP irony detection, multi-agent debate (vLLM): K=5 agents, 3 rounds in total (the initial
independent round included). Soft label = [1 - fraction Yes, fraction Yes] over the K final answers.
Seeding: enable_prefix_caching=False and one n=1 request per agent and round, each with its own seed.

Usage:
  python3 generation/lewidi_debate.py --model models/Qwen3-32B --data data/MP_test.json \
      --seed 43 --out results/p1/C_mp_qwen_debateF_k5_v2_s43.npz
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

EPS = 1e-9; K = 5; L_ROUNDS = 3; TEMP = 0.7; TOP_P = 0.95; GEN_MAX = 200

INIT = (
    'Read the post and reason briefly step by step about whether it is ironic, '
    'then end your answer with a line exactly of the form "Answer: Yes" or "Answer: No".\n\n'
    'Post: "{x}"'
)
DEBATE = (
    'Read the post and decide whether it is ironic.\n\nPost: "{x}"\n\n'
    'These are the answers to this question from other agents:\n\n{others}\n\n'
    'Using the reasoning from these other agents as additional information, '
    'examine your own reading and theirs step by step, and give an updated answer. '
    'End your answer with a line exactly of the form "Answer: Yes" or "Answer: No".'
)


def per_sample_seed(run_seed, item_idx, sample_k):
    raw = hashlib.md5(f"{run_seed}:{item_idx}:{sample_k}".encode()).digest()
    return int.from_bytes(raw[:4], 'little') & 0x7FFF_FFFF


def gen_k_per_item(llm, prompts, K, run_seed, temperature, top_p, max_tokens, offset=0):
    N = len(prompts)
    flat_p = [prompts[i] for i in range(N) for _ in range(K)]
    flat_sp = [SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              seed=per_sample_seed(run_seed, i + offset, k))
               for i in range(N) for k in range(K)]
    outs = llm.generate(_ADAPT['f'](flat_p), flat_sp)
    return [[outs[i*K+k].outputs[0].text.strip() for k in range(K)] for i in range(N)]


def gen_debate_round(llm, flat_prompts, run_seed, round_idx, temperature, top_p, max_tokens):
    n_reqs = len(flat_prompts)
    flat_sp = [SamplingParams(n=1, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
                              seed=per_sample_seed(run_seed, j, round_idx))
               for j in range(n_reqs)]
    outs = llm.generate(_ADAPT['f'](flat_prompts), flat_sp)
    return [o.outputs[0].text.strip() for o in outs]


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
    print(f'[lewidi-debate] model={Path(args.model).name} N={N} K={K} L={L_ROUNDS} seed={args.seed}', flush=True)

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

    # Round 0: init K samples per item
    init_prompts = [chat(INIT.format(x=x)) for x in xs]
    answers = gen_k_per_item(llm, init_prompts, K, args.seed, TEMP, TOP_P, GEN_MAX, offset=0)

    # Rounds 1..L-1: debate
    for round_idx in range(1, L_ROUNDS):
        fp = []
        for i in range(N):
            for a in range(K):
                others = '\n\n'.join(f"Agent {j+1}'s answer:\n{answers[i][j]}" for j in range(K) if j != a)
                fp.append(chat(DEBATE.format(x=xs[i], others=others)))
        texts = gen_debate_round(llm, fp, args.seed, round_idx, TEMP, TOP_P, GEN_MAX)
        answers = [[texts[i*K+a] for a in range(K)] for i in range(N)]

    dt = time.time() - t0
    p = np.zeros((N, 2), dtype=np.float32)
    for i in range(N):
        yes = sum(1 for a in range(K) if parse_yes(answers[i][a]) == 1)
        frac = yes / K; p[i] = [1-frac, frac]

    p_human = np.stack([np.array(r['p_human'], dtype=np.float32) for r in recs])
    ids = np.array([str(r['id']) for r in recs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, p=p, p_human=p_human, ids=ids)
    print(f'[lewidi-debate DONE] N={N} K={K} L={L_ROUNDS} {dt:.0f}s -> {args.out}', flush=True)

    from scipy.stats import spearmanr
    def binent(pp): a = np.clip(pp, EPS, 1-EPS); return (-(a*np.log2(a)+(1-a)*np.log2(1-a))).mean(axis=1)
    rho, _ = spearmanr(binent(p.astype(np.float64)), binent(p_human.astype(np.float64)))
    print(f'[sanity] rhoH={rho:+.4f}', flush=True)

if __name__ == '__main__': main()
