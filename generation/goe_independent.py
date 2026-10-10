#!/usr/bin/env python3
"""GoEmotions, independent sampling (vLLM): K samples per item, soft label = fraction of samples
that list each emotion. Formats: plain listing, CoT listing (self-consistency), and per-label Yes/No.

Seeding:
  - enable_prefix_caching=False, so that cached tokens are never reused across requests
    that share a prompt but have different seeds;
  - flat batch of N*K requests with n=1, each with its own seed = hash(run_seed, item_idx, k).
  Different run seeds give different outputs, the K samples of an item are independent, and the
  same run seed is reproducible up to the nondeterminism of batched GPU inference.

Output npz:
  p  : (N, 28) float32, soft probabilities over the 28 GoEmotions labels
  ids: (N,)    str

Usage:
  python3 generation/goe_independent.py \
    --format plain --k 5 \
    --model  models/Qwen3-32B \
    --data   data/goe_test_prepared.json \
    --sc_ref ref/C_sc_probs.npz \
    --seed   43 \
    --out    results/p1/C_goe_qwen_plain_k5_v2_s43.npz
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

K = 5
TEMP = 0.7
TOP_P = 0.95
GEN_MAX = 256

COT_PROMPT = (
    'Read the utterance and reason briefly step by step about which emotions it expresses, '
    'then end your answer with a line exactly of the form:\n'
    'Emotions: <comma-separated subset of [{labels}], or "neutral">\n\n'
    'Utterance: "{x}"'
)
PLAIN_PROMPT = (
    'List the emotions expressed in the utterance. Answer with a single line exactly of the form:\n'
    'Emotions: <comma-separated subset of [{labels}], or "neutral">\n\n'
    'Utterance: "{x}"'
)
YESNO_PROMPT = (
    'For the utterance below, decide for EACH candidate emotion whether it is expressed.\n'
    'Answer with one line per emotion in the exact form "<emotion>: Yes" or "<emotion>: No", '
    'covering ALL of these emotions: [{labels}].\n\n'
    'Utterance: "{x}"'
)
PROMPTS = {'cot': COT_PROMPT, 'plain': PLAIN_PROMPT, 'yesno': YESNO_PROMPT}


def per_sample_seed(run_seed: int, item_idx: int, sample_k: int) -> int:
    raw = hashlib.md5(f"{run_seed}:{item_idx}:{sample_k}".encode()).digest()
    return int.from_bytes(raw[:4], 'little') & 0x7FFF_FFFF


def gen_k_per_item(llm: LLM, prompts: list, K: int, run_seed: int,
                   temperature: float, top_p: float, max_tokens: int) -> list:
    """Flat-batch n=1 generation. Returns list[list[str]]: outputs[i][k]."""
    N = len(prompts)
    flat_prompts = [prompts[i] for i in range(N) for _ in range(K)]
    flat_sp = [
        SamplingParams(n=1, temperature=temperature, top_p=top_p,
                       max_tokens=max_tokens, seed=per_sample_seed(run_seed, i, k))
        for i in range(N) for k in range(K)
    ]
    outs = llm.generate(_ADAPT['f'](flat_prompts), flat_sp)
    return [[outs[i * K + k].outputs[0].text for k in range(K)] for i in range(N)]


def parse_listing(text: str, emotions: list) -> set:
    m = re.search(r'Emotions?\s*:\s*(.+)', text, re.IGNORECASE)
    span = m.group(1) if m else text
    low = span.lower()
    return {e for e in emotions if re.search(r'\b' + re.escape(e) + r'\b', low)}


def parse_yesno(text: str, emotions: list) -> set:
    low = text.lower()
    result = set()
    for e in emotions:
        m = re.search(re.escape(e) + r'\s*:\s*(yes|no)', low)
        if m and m.group(1) == 'yes':
            result.add(e)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--format', required=True, choices=['cot', 'plain', 'yesno'])
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--sc_ref', required=True, help='npz with the 1,200 subset ids (ref/C_sc_probs.npz)')
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--k', type=int, default=K)
    ap.add_argument('--gpu_mem', type=float, default=0.90)
    args = ap.parse_args()

    K_run = args.k
    data = json.load(open(args.data))
    emotions = data['emotions']
    records = data['records']

    # Restrict to the 1,200-item evaluation subset
    sc_ref = np.load(args.sc_ref, allow_pickle=True)
    sub_ids = [str(x) for x in sc_ref['ids']]
    id2rec = {str(r['id']): r for r in records}
    recs = [id2rec[i] for i in sub_ids if i in id2rec]
    if args.limit:
        recs = recs[:args.limit]
    N = len(recs)

    print(f'[goe-independent] format={args.format} model={Path(args.model).name} '
          f'N={N} K={K_run} seed={args.seed}', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    plabels = ', '.join(emotions)

    def chat(u):
        m = [{'role': 'user', 'content': u}]
        try:
            return tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    _chat_raw = chat

    def chat(u):
        # vLLM adds BOS itself when tokenizing a text prompt; drop a BOS the template rendered
        # as text (Mistral would otherwise get two). No effect on Qwen/Yi.
        s = _chat_raw(u)
        b = tok.bos_token
        return s[len(b):] if b and s.startswith(b) else s

    xs = [r['text'] for r in recs]
    PROMPT_TMPL = PROMPTS[args.format]
    prompts = [chat(PROMPT_TMPL.format(labels=plabels, x=x)) for x in xs]

    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, seed=args.seed,
              enforce_eager=True,
              enable_prefix_caching=False)  # required for per-request seeds
    _ADAPT['f'] = make_prompt_adapter(llm.get_tokenizer(), args.model, [chat('probe')])

    t0 = time.time()
    all_outputs = gen_k_per_item(llm, prompts, K_run, args.seed, TEMP, TOP_P, GEN_MAX)

    parse = parse_yesno if args.format == 'yesno' else parse_listing
    p = np.zeros((N, len(emotions)), dtype=np.float32)
    for i, outputs_i in enumerate(all_outputs):
        counts = np.zeros(len(emotions))
        for text in outputs_i:
            for e in parse(text, emotions):
                counts[emotions.index(e)] += 1
        p[i] = counts / K_run

    dt = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ids = np.array([str(r['id']) for r in recs])
    np.savez(args.out, p=p, ids=ids)
    print(f'[goe-independent DONE] format={args.format} N={N} K={K_run} '
          f'{dt:.0f}s ({N/dt:.2f} ex/s) -> {args.out}', flush=True)

    # Sanity check: rhoH against the human soft labels stored in the data file
    p_human_list = [r.get('p_human') for r in recs]
    if all(h is not None and len(h) == len(emotions) for h in p_human_list):
        from scipy.stats import spearmanr
        eps = 1e-9
        def bent(q):
            q = np.clip(q.astype(np.float64), eps, 1 - eps)
            return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        p_human = np.stack([np.array(h, dtype=np.float64) for h in p_human_list])
        Hm = bent(p).mean(axis=1); Hh = bent(p_human).mean(axis=1)
        rhoH, _ = spearmanr(Hm, Hh)
        print(f'[sanity] rhoH={rhoH:+.4f}', flush=True)


if __name__ == '__main__':
    main()
