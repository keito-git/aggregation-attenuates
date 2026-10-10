#!/usr/bin/env python3
"""GoEmotions, zero-shot classification and CCD (context generation), HuggingFace transformers.

  S0 (zero-shot): per-label Yes/No classification, p(Yes) from next-token log-probabilities -> p_base
  CCD: for the N_HYP=4 most likely labels under S0, a proposer writes one context sentence that would
       make the utterance express that label; every label is re-classified given each context; the
       context-conditioned predictions are mixed with weights
       softmax_j( log p(x|c_j) - log q(c_j|x) + log p(c_j) )                              -> p_ccd
Progress is checkpointed next to --out (<out>.ckpt.npz) and resumed automatically.

Output npz: p_base (N,28), p_ccd (N,28), ids (N,)

Usage:
  python3 generation/goe_ccd.py \
    --model  models/Qwen3-32B \
    --data   data/goe_test_prepared.json \
    --sc_ref ref/C_sc_probs.npz \
    --seed   43 \
    --out    results/p1/C_goe_qwen_ccd_v2_s43.npz
"""
import os, json, time, argparse, hashlib
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('FLASHINFER_CUDA_ARCH_LIST', '9.0')
os.environ.setdefault('VLLM_USE_FLASHINFER_SAMPLER', '0')

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MAX_NEW_TOKENS = 64   # proposer generation length
N_HYP = 4            # number of hypotheses per example

# Prompts
CLASSIFY_TMPL = 'Classify: does the utterance "{x}" express {label}? Answer with a single word: Yes or No.'
CLASSIFY_CTX_TMPL = 'Context: {c}\nClassify: does the utterance "{x}" express {label}? Answer with a single word: Yes or No.'
PROPOSER_TMPL = 'Write one plausible sentence of context that could come before the utterance and make it clearly express {label}. Write only the context sentence.\nUtterance: "{x}"'


# ── Per-sample seeds ─────────────────────────────────────────────────────────

def per_sample_seed(run_seed: int, item_idx: int, sample_k: int) -> int:
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

    @torch.no_grad()
    def generate_one(self, prompt_ids: list, seed_val: int) -> str:
        """Greedy generation for one prompt (proposer); the torch seed is set per call."""
        torch.manual_seed(seed_val)
        pad = self.tok.pad_token_id or 0
        input_ids = torch.tensor([prompt_ids], device=self.device)
        attn = torch.ones_like(input_ids)
        gen = self.model.generate(
            input_ids, attention_mask=attn,
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1,
            pad_token_id=pad,
        )
        new = gen[0, input_ids.shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip()

    @torch.no_grad()
    def logprob_raw(self, prefix_ids: list, target_ids: list) -> float:
        """Teacher-forcing logprob sum log p(target | prefix)."""
        if len(target_ids) == 0:
            return 0.0
        ids = torch.tensor([prefix_ids + target_ids], device=self.device)
        out = self.model(ids)
        logits = out.logits[0]
        n_pre = len(prefix_ids)
        if n_pre == 0:
            if len(target_ids) < 2:
                return 0.0
            shift = logits[0: len(target_ids) - 1, :]
            tgt = target_ids[1:]
        else:
            shift = logits[n_pre - 1: n_pre + len(target_ids) - 1, :]
            tgt = target_ids
        lp = torch.log_softmax(shift.float(), dim=-1)
        tgt_t = torch.tensor(tgt, device=self.device)
        sel = lp[torch.arange(len(tgt), device=self.device), tgt_t]
        return float(sel.sum().item())


# ── CCD per-example processing ───────────────────────────────────────────────

def process_example(lm: LM, emotions: list, x: str, item_idx: int, run_seed: int):
    """Zero-shot (S0) and CCD predictions for one example."""
    K = len(emotions)
    # S0: zero-shot classification (deterministic logprob)
    p0 = lm.classify_batch([CLASSIFY_TMPL.format(x=x, label=lbl) for lbl in emotions])

    # Top-N_HYP candidate labels
    top = list(np.argsort(-p0)[:N_HYP])
    cand_labels = [emotions[i] for i in top]

    # Proposer: generate one context per candidate label (per-sample seeded)
    contexts = []
    for j, lbl in enumerate(cand_labels):
        seed_j = per_sample_seed(run_seed, item_idx, j)
        prop_prompt_ids = lm._chat_ids(PROPOSER_TMPL.format(x=x, label=lbl))
        c = lm.generate_one(prop_prompt_ids, seed_j)
        contexts.append(c)

    # Context-conditioned classification: N_HYP × 28
    p_ctx = np.zeros((N_HYP, K))
    for j, c in enumerate(contexts):
        p_ctx[j] = lm.classify_batch(
            [CLASSIFY_CTX_TMPL.format(c=c, x=x, label=lbl) for lbl in emotions]
        )

    # Scorer: s_j = logp(x|c_j) - logq(c_j|x) + logp(c_j)
    x_ids = lm.tok.encode(" " + x, add_special_tokens=False)
    bos_id = lm.tok.bos_token_id
    bos = [bos_id] if bos_id is not None else []
    s = np.zeros(N_HYP)
    for j, c in enumerate(contexts):
        c_ids = lm.tok.encode(c, add_special_tokens=False)
        c_prefix = bos + c_ids
        logp_x_given_c = lm.logprob_raw(c_prefix, x_ids)
        prop_ids = lm._chat_ids(PROPOSER_TMPL.format(x=x, label=cand_labels[j]))
        logq_c_given_x = lm.logprob_raw(prop_ids, c_ids)
        logp_c = lm.logprob_raw(bos, c_ids)
        s[j] = logp_x_given_c - logq_c_given_x + logp_c

    w = np.exp(s - s.max())
    w = w / w.sum()
    p_ccd = (w[:, None] * p_ctx).sum(axis=0)

    return p0.astype(np.float32), p_ccd.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model',  required=True)
    ap.add_argument('--data',   required=True)
    ap.add_argument('--sc_ref', required=True, help='npz with the 1,200 subset ids (ref/C_sc_probs.npz)')
    ap.add_argument('--seed',   type=int, default=43)
    ap.add_argument('--out',    required=True)
    ap.add_argument('--limit',  type=int, default=0)
    ap.add_argument('--verbose', type=int, default=0)
    args = ap.parse_args()

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

    print(f'[goe-ccd] model={Path(args.model).name} N={N} N_HYP={N_HYP} seed={args.seed}',
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

    p_base = np.full((N, len(emotions)), np.nan, dtype=np.float32)
    p_ccd  = np.full((N, len(emotions)), np.nan, dtype=np.float32)
    start  = 0

    if ckpt_path.exists() and not args.limit:
        z = np.load(ckpt_path)
        if z['p_base'].shape[0] == N:
            p_base = z['p_base']; p_ccd = z['p_ccd']
            start = int(z['done'])
            print(f'  Resume from {start}/{N}', flush=True)

    t0 = time.time()
    for i in range(start, N):
        vb = i < args.verbose
        pb, pc = process_example(lm, emotions, records[i]['text'], i, args.seed)
        p_base[i] = pb; p_ccd[i] = pc
        if vb:
            print(f'  [{i}] ccd top5: '
                  f'{[(emotions[j], round(float(p_ccd[i][j]),3)) for j in np.argsort(-p_ccd[i])[:5]]}',
                  flush=True)
        if (i + 1) % 20 == 0 or i == N - 1:
            r = (i + 1 - start) / (time.time() - t0)
            eta = (N - i - 1) / r / 60 if r > 0 else 99999
            print(f'  [{i+1}/{N}] {r:.2f} ex/s ETA {eta:.0f}min', flush=True)
            if not args.limit:
                np.savez(ckpt_path, p_base=p_base, p_ccd=p_ccd, done=i + 1)

    ids = np.array([str(r['id']) for r in records])
    np.savez(args.out, p_base=p_base, p_ccd=p_ccd, ids=ids)
    print(f'[goe-ccd DONE] N={N} -> {args.out}', flush=True)

    # Sanity check: rhoH against the human soft labels stored in the data file
    from scipy.stats import spearmanr
    eps = 1e-9
    p_human_list = [r.get('p_human') for r in records]
    if all(h is not None and len(h) == len(emotions) for h in p_human_list):
        p_human = np.stack([np.array(h, dtype=np.float64) for h in p_human_list])
        def bent(q):
            q = np.clip(q.astype(np.float64), eps, 1 - eps)
            return -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
        rhoH, _ = spearmanr(bent(p_ccd).mean(axis=1), bent(p_human).mean(axis=1))
        rhoH_base, _ = spearmanr(bent(p_base).mean(axis=1), bent(p_human).mean(axis=1))
        print(f'[sanity] rhoH(ccd)={rhoH:+.4f}  rhoH(base/zeroshot)={rhoH_base:+.4f}', flush=True)


if __name__ == '__main__':
    main()
