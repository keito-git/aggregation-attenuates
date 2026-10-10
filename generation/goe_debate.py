#!/usr/bin/env python3
"""GoEmotions, multi-agent debate (vLLM): K=5 agents, L rounds in total (default 3, the initial
independent round included). In every later round each agent sees the other agents' previous answers
and gives an updated answer. Soft label = fraction of the K final answers that list each emotion.

Seeding: enable_prefix_caching=False and one n=1 request per agent and round, each with its own seed.

Output npz: p (N,28), p_human (N,28), ids (N,)

Usage:
  python3 generation/goe_debate.py \
    --model  models/Qwen3-32B \
    --data   data/goe_test_prepared.json \
    --sc_ref ref/C_sc_probs.npz \
    --seed   43 [--rounds 3] \
    --out    results/p1/C_goe_qwen_debateF_k5_v2_s43.npz
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
L_ROUNDS = 3
TEMP = 0.7
TOP_P = 0.95
GEN_MAX = 256

INIT_TMPL = (
    'Read the utterance and reason briefly step by step about which emotions it expresses, '
    'then end your answer with a line exactly of the form:\n'
    'Emotions: <comma-separated subset of [{labels}], or "neutral">\n\n'
    'Utterance: "{x}"'
)
DEBATE_TMPL = (
    'Read the utterance and decide which emotions it expresses.\n\n'
    'Utterance: "{x}"\n\n'
    'These are the answers to this question from other agents:\n\n{others}\n\n'
    'Using the reasoning from these other agents as additional information, '
    'examine your own reading and theirs step by step, and give an updated answer. '
    'End your answer with a line exactly of the form:\n'
    'Emotions: <comma-separated subset of [{labels}], or "neutral">'
)


def per_sample_seed(run_seed: int, item_idx: int, sample_k: int) -> int:
    raw = hashlib.md5(f"{run_seed}:{item_idx}:{sample_k}".encode()).digest()
    return int.from_bytes(raw[:4], 'little') & 0x7FFF_FFFF


def gen_k_per_item(llm: LLM, prompts: list, K: int, run_seed: int,
                   temperature: float, top_p: float, max_tokens: int,
                   seed_offset: int = 0) -> list:
    """Flat-batch n=1 per-sample-seeded generation. Returns list[list[str]]."""
    N = len(prompts)
    flat_prompts = [prompts[i] for i in range(N) for _ in range(K)]
    flat_sp = [
        SamplingParams(n=1, temperature=temperature, top_p=top_p,
                       max_tokens=max_tokens,
                       seed=per_sample_seed(run_seed, i + seed_offset * 10000, k))
        for i in range(N) for k in range(K)
    ]
    outs = llm.generate(_ADAPT['f'](flat_prompts), flat_sp)
    return [[outs[i * K + k].outputs[0].text.strip() for k in range(K)] for i in range(N)]


def gen_debate_round(llm: LLM, flat_prompts: list, run_seed: int, round_idx: int,
                     temperature: float, top_p: float, max_tokens: int) -> list:
    """Debate round: N*K unique prompts, each n=1 with unique per-request seed."""
    n_reqs = len(flat_prompts)
    flat_sp = [
        SamplingParams(n=1, temperature=temperature, top_p=top_p,
                       max_tokens=max_tokens,
                       seed=per_sample_seed(run_seed, j, round_idx))
        for j in range(n_reqs)
    ]
    outs = llm.generate(_ADAPT['f'](flat_prompts), flat_sp)
    return [o.outputs[0].text.strip() for o in outs]


def parse_emotions(text: str, emotions: list) -> set:
    m = re.search(r'Emotions?\s*:\s*(.+)', text, re.IGNORECASE)
    span = m.group(1) if m else text
    low = span.lower()
    return {e for e in emotions if re.search(r'\b' + re.escape(e) + r'\b', low)}


def human_vec(r: dict, emotions: list, e_idx: dict) -> np.ndarray:
    v = np.zeros(len(emotions), dtype=np.float32)
    hl = r.get('human_soft') or r.get('p_human') or r.get('soft_labels')
    if isinstance(hl, dict):
        for e, val in hl.items():
            if e in e_idx:
                v[e_idx[e]] = val
    elif isinstance(hl, (list, np.ndarray)) and len(hl) == len(emotions):
        v = np.array(hl, dtype=np.float32)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--sc_ref', required=True)
    ap.add_argument('--seed', type=int, default=43)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--gpu_mem', type=float, default=0.90)
    ap.add_argument("--rounds", type=int, default=L_ROUNDS,
                    help="total rounds incl. the initial one (default = main setting; varied in the dose-response runs)")
    args = ap.parse_args()

    data = json.load(open(args.data))
    emotions = data['emotions']
    records = data['records']

    sc_ref = np.load(args.sc_ref, allow_pickle=True)
    sub_ids = [str(x) for x in sc_ref['ids']]
    id2rec = {str(r['id']): r for r in records}
    recs = [id2rec[i] for i in sub_ids if i in id2rec]
    if args.limit:
        recs = recs[:args.limit]
    N = len(recs)
    e_idx = {e: j for j, e in enumerate(emotions)}
    plabels = ', '.join(emotions)

    print(f'[goe-debate] model={Path(args.model).name} N={N} K={K} L={args.rounds} '
          f'seed={args.seed}', flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)

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

    llm = LLM(model=args.model, dtype='bfloat16',
              gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, seed=args.seed,
              enforce_eager=True,
              enable_prefix_caching=False)  # required for per-request seeds
    _ADAPT['f'] = make_prompt_adapter(llm.get_tokenizer(), args.model, [chat('probe')])

    t0 = time.time()

    # Round 0: init (K samples per item via flat-batch)
    init_prompts = [chat(INIT_TMPL.format(labels=plabels, x=x)) for x in xs]
    answers = gen_k_per_item(llm, init_prompts, K, args.seed, TEMP, TOP_P, GEN_MAX,
                             seed_offset=0)

    # Rounds 1..rounds-1: debate
    for round_idx in range(1, args.rounds):
        flat_prompts = []
        for i in range(N):
            for a in range(K):
                others = '\n\n'.join(
                    f"Agent {j+1}'s answer:\n{answers[i][j]}"
                    for j in range(K) if j != a
                )
                flat_prompts.append(
                    chat(DEBATE_TMPL.format(labels=plabels, x=xs[i], others=others))
                )
        texts = gen_debate_round(llm, flat_prompts, args.seed, round_idx,
                                 TEMP, TOP_P, GEN_MAX)
        answers = [[texts[i * K + a] for a in range(K)] for i in range(N)]

    # Final soft probabilities
    p = np.zeros((N, len(emotions)), dtype=np.float32)
    for i in range(N):
        counts = np.zeros(len(emotions))
        for a in range(K):
            for e in parse_emotions(answers[i][a], emotions):
                counts[e_idx[e]] += 1
        p[i] = counts / K

    dt = time.time() - t0
    p_human = np.stack([human_vec(r, emotions, e_idx) for r in recs])
    ids = np.array([str(r['id']) for r in recs])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, p=p, p_human=p_human, ids=ids)
    print(f'[goe-debate DONE] N={N} K={K} L={args.rounds} {dt:.0f}s -> {args.out}',
          flush=True)

    from scipy.stats import spearmanr
    eps = 1e-9
    def bent(q): q = np.clip(q.astype(np.float64), eps, 1-eps); return -(q*np.log2(q)+(1-q)*np.log2(1-q))
    rhoH, _ = spearmanr(bent(p).mean(axis=1), bent(p_human).mean(axis=1))
    print(f'[sanity] rhoH={rhoH:+.4f}', flush=True)


if __name__ == '__main__':
    main()
