#!/usr/bin/env python3
"""GoEmotions, input-clarification ensembling (Hou et al., ICML 2024), HuggingFace transformers.

For each utterance, K_CLAR=5 clarified restatements are sampled (each with its own torch seed derived
from (run_seed, item_idx, k)); for each clarification, every label is classified with a Yes/No prompt
and p(Yes) is read from the next-token log-probabilities; the soft label is the mean over the K_CLAR
clarifications. Progress is checkpointed next to --out (<out>.ckpt.npz) and resumed automatically.

Output npz: p_hou (N,28), p_clar (N,K_CLAR,28), ids (N,), tokens (N,)

Usage:
  python3 generation/goe_clarification.py \
    --model  models/Qwen3-32B \
    --data   data/goe_test_prepared.json \
    --sc_ref ref/C_sc_probs.npz \
    --seed   43 \
    --out    results/p1/C_goe_qwen_hou_v2_s43.npz
"""
import os, json, time, argparse, hashlib
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
# same environment as the vLLM scripts (not used by transformers)
os.environ.setdefault('FLASHINFER_CUDA_ARCH_LIST', '9.0')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

GEN_MAX = 64        # clarification generation max tokens
TEMPERATURE = 0.7   # sampling temperature (same as the vLLM scripts)
K_CLAR = 5          # number of clarifications per example (compute-matched)

# Prompts
CLARIFY_TMPL = (
    'The following utterance may be ambiguous about what the speaker means and feels. '
    'Write one clarified restatement of the utterance that makes the speaker\'s intended '
    'meaning and tone explicit. Write only the clarified restatement.\n'
    'Utterance: "{x}"'
)
CLASSIFY_CLAR_TMPL = (
    'Clarification of the utterance: {c}\n'
    'Classify: does the utterance "{x}" express {label}? '
    'Answer with a single word: Yes or No.'
)


# ── Per-sample seeds ─────────────────────────────────────────────────────────

def per_sample_seed(run_seed: int, item_idx: int, sample_k: int) -> int:
    """Deterministic per-sample seed: hash(run_seed, item_idx, sample_k)."""
    raw = hashlib.md5(f"{run_seed}:{item_idx}:{sample_k}".encode()).digest()
    return int.from_bytes(raw[:4], 'little') & 0x7FFF_FFFF


# ── Language model wrapper ───────────────────────────────────────────────────

class LM:
    def __init__(self, model, tok, device):
        self.model, self.tok, self.device = model, tok, device
        self.yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        self.no_id  = tok.encode("No",  add_special_tokens=False)[0]

    def _chat_ids(self, user_text):
        # transformers 5.x: apply_chat_template(tokenize=True) no longer returns a plain
        # list of ids; render the template as text, then tokenize.
        msgs = [{"role": "user", "content": user_text}]
        try:
            text = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False
            )
        except TypeError:
            text = self.tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
        return list(self.tok(text, add_special_tokens=False)["input_ids"])

    @torch.no_grad()
    def classify_batch(self, prompts):
        """p(Yes) = softmax([logP(Yes), logP(No)])[0] for each prompt.
        Right padding + attention mask; the logits are read at the last real position."""
        seqs = [self._chat_ids(p) for p in prompts]
        maxlen = max(len(s) for s in seqs)
        pad = self.tok.pad_token_id or 0
        input_ids, attn, last_pos = [], [], []
        for s in seqs:
            padlen = maxlen - len(s)
            input_ids.append(s + [pad] * padlen)
            attn.append([1] * len(s) + [0] * padlen)
            last_pos.append(len(s) - 1)
        input_ids = torch.tensor(input_ids, device=self.device)
        attn = torch.tensor(attn, device=self.device)
        out = self.model(input_ids, attention_mask=attn)
        idx = torch.tensor(last_pos, device=self.device)
        logits = out.logits[torch.arange(len(seqs), device=self.device), idx, :]
        lp = torch.log_softmax(logits.float(), dim=-1)
        two = torch.stack([lp[:, self.yes_id], lp[:, self.no_id]], dim=-1)
        p_yes = torch.softmax(two, dim=-1)[:, 0]
        return p_yes.cpu().numpy()

# ── Clarification generation ──────────────────────────────────────────────────

def sample_clarifications(lm: LM, x: str, k: int, item_idx: int, run_seed: int):
    """Generate k clarifications with per-sample seeds."""
    ids = lm._chat_ids(CLARIFY_TMPL.format(x=x))
    pad = lm.tok.pad_token_id or lm.tok.eos_token_id
    texts, toks = [], []
    for j in range(k):
        seed_j = per_sample_seed(run_seed, item_idx, j)
        torch.manual_seed(seed_j)
        input_ids = torch.tensor([ids], device=lm.device)
        attn = torch.ones_like(input_ids)
        with torch.no_grad():
            gen = lm.model.generate(
                input_ids, attention_mask=attn, max_new_tokens=GEN_MAX,
                do_sample=True, temperature=TEMPERATURE, top_p=0.95,
                pad_token_id=pad
            )
        new = gen[0, input_ids.shape[1]:]
        texts.append(lm.tok.decode(new, skip_special_tokens=True).strip())
        toks.append(len(ids) + int((new != pad).sum()))
    return texts, toks


def hou_example(lm: LM, emotions: list, x: str, item_idx: int, run_seed: int):
    """Run clarification ensembling for one example."""
    clars, gtoks = sample_clarifications(lm, x, K_CLAR, item_idx, run_seed)
    total = sum(gtoks)
    p_clar = np.zeros((K_CLAR, len(emotions)), dtype=np.float32)
    for j, c in enumerate(clars):
        prompts = [CLASSIFY_CLAR_TMPL.format(c=c, x=x, label=lbl) for lbl in emotions]
        total += sum(len(lm._chat_ids(p)) + 1 for p in prompts)
        p_clar[j] = lm.classify_batch(prompts)
    p_mean = p_clar.mean(axis=0)
    return p_mean, p_clar, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  required=True, help='model directory')
    ap.add_argument('--data',   required=True, help='goe_test_prepared.json')
    ap.add_argument('--sc_ref', required=True, help='npz with the 1,200 subset ids (ref/C_sc_probs.npz)')
    ap.add_argument('--seed',   type=int, default=43)
    ap.add_argument('--out',    required=True)
    ap.add_argument('--limit',  type=int, default=0)
    ap.add_argument('--verbose', type=int, default=0)
    ap.add_argument('--gpu_mem_fraction', type=float, default=0.9)
    args = ap.parse_args()

    # Set global seed (per-sample seeding also applied per clarification)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    data = json.load(open(args.data))
    emotions = data['emotions']

    # Restrict to the 1,200-item evaluation subset
    sc_ref = np.load(args.sc_ref, allow_pickle=True)
    sub_ids = [str(x) for x in sc_ref['ids']]
    id2rec = {str(r['id']): r for r in data['records']}
    records = [id2rec[i] for i in sub_ids if i in id2rec]
    if args.limit:
        records = records[:args.limit]
    N = len(records)

    print(f'[goe-clarification] model={Path(args.model).name} N={N} K={K_CLAR} seed={args.seed}',
          flush=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map={'': device}
    ).eval()
    lm = LM(model, tok, device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix('.ckpt.npz')

    p_hou  = np.full((N, len(emotions)), np.nan, dtype=np.float32)
    p_clar = np.full((N, K_CLAR, len(emotions)), np.nan, dtype=np.float32)
    tokens = np.zeros(N, dtype=np.int64)
    start = 0

    # Resume from checkpoint
    if ckpt_path.exists() and not args.limit:
        z = np.load(ckpt_path)
        if z['p_hou'].shape[0] == N:
            p_hou = z['p_hou']; p_clar = z['p_clar']; tokens = z['tokens']
            start = int(z['done'])
            print(f'  Resume from {start}/{N}', flush=True)

    t0 = time.time()
    for i in range(start, N):
        vb = i < args.verbose
        pm, pc, total = hou_example(lm, emotions, records[i]['text'], i, args.seed)
        p_hou[i] = pm; p_clar[i] = pc; tokens[i] = total
        if vb:
            print(f'  [{i}] tok={total} top5: '
                  f'{[(emotions[j], round(float(p_hou[i][j]),3)) for j in np.argsort(-p_hou[i])[:5]]}',
                  flush=True)
        if (i + 1) % 20 == 0 or i == N - 1:
            r = (i + 1 - start) / (time.time() - t0)
            eta = (N - i - 1) / r / 60 if r > 0 else 99999
            print(f'  [{i+1}/{N}] {r:.2f} ex/s ETA {eta:.0f}min mean_tok={tokens[start:i+1].mean():.0f}',
                  flush=True)
            if not args.limit:
                np.savez(ckpt_path, p_hou=p_hou, p_clar=p_clar, tokens=tokens, done=i + 1)

    ids = np.array([str(r['id']) for r in records])
    np.savez(args.out, p_hou=p_hou, p_clar=p_clar, ids=ids, tokens=tokens)
    print(f'[goe-clarification DONE] N={N} mean_tokens/ex={tokens.mean():.0f} -> {args.out}',
          flush=True)

    # Sanity check: rhoH against the human soft labels stored in the data file
    from scipy.stats import spearmanr
    eps = 1e-9
    p_human_list = [r.get('p_human') for r in records]
    if all(h is not None and len(h) == len(emotions) for h in p_human_list):
        p_human = np.stack([np.array(h, dtype=np.float64) for h in p_human_list])
        def bent(q):
            q = np.clip(q.astype(np.float64), eps, 1 - eps)
            return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        Hm = bent(p_hou).mean(axis=1)
        Hh = bent(p_human).mean(axis=1)
        rhoH, _ = spearmanr(Hm, Hh)
        print(f'[sanity] rhoH={rhoH:+.4f}', flush=True)


if __name__ == '__main__':
    main()
