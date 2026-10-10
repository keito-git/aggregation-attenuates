# Aggregation Attenuates the Human Disagreement Signal in Compute-Matched Multi-Agent LLM Soft-Label Prediction

Code for the paper of the same title (Neurocomputing, under review). The code generates soft-label
predictions with compute-matched single-agent and multi-agent LLM methods on GoEmotions and LeWiDi-MP,
and computes every number, table and result figure reported in the paper.

## Contents

```
generation/   GPU scripts that produce the soft-label predictions (one .npz per method x model x seed)
  goe_independent.py      GoEmotions, independent sampling: plain listing, CoT / self-consistency, Yes/No; K samples
  goe_debate.py           GoEmotions, multi-agent debate (K=5 agents, L rounds; default L=3)
  goe_moa.py              GoEmotions, Mixture-of-Agents (K=5 agents per layer, L layers; default L=3)
  goe_clarification.py    GoEmotions, input-clarification ensembling (transformers)
  goe_ccd.py              GoEmotions, zero-shot classification and CCD context generation (transformers)
  lewidi_independent.py   LeWiDi-MP irony, independent sampling (K=5)
  lewidi_debate.py        LeWiDi-MP irony, debate (K=5, 3 rounds)
  lewidi_moa.py           LeWiDi-MP irony, Mixture-of-Agents (K=5, 3 layers)
  prompt_encoding.py      chat-marker-safe prompt encoding for vLLM (see "Chat templates and BOS")
  checks/                 CPU checks of the prompt encoding (Mistral against mistral_common; all models)
analysis/     CPU analysis
  build_goe_reference.py      1,200-item GoEmotions reference (ids, human soft labels, entropy strata)
  human_reliability_ceiling.py split-half / Spearman-Brown ceiling for rho_H
  final_analysis.py           all metrics, bootstrap CIs, Holm-corrected contrasts, dose-response test;
                              writes numbers.tex (LaTeX macros), final_numbers.json and two table bodies
  make_figures.py             result figures from final_numbers.json
  make_prompts_appendix.py    verbatim prompt appendix, read from the prompt constants in generation/
  inventory_results.py        completeness and validity check of results/
  output_diagnostics.py       plausibility checks of the outputs (diagnostic only; not used in the paper)
data/         dataset preparation and download instructions (no dataset files are included)
ref/          C_sc_probs.npz: ids of the 1,200-item GoEmotions evaluation subset
scripts/      setup_env.sh, run_main.sh, run_dose_response.sh, run_analysis.sh, flashinfer_sm120.patch
results/      generated model outputs analysed in the paper (see results/README.md)
```

## Models

All models are loaded from Hugging Face in bfloat16:

| family | repository id |
|---|---|
| Qwen3-32B | `Qwen/Qwen3-32B` |
| Mistral-24B | `mistralai/Mistral-Small-24B-Instruct-2501` |
| Yi-1.5-34B | `01-ai/Yi-1.5-34B-Chat` |

The repositories were downloaded from their default branch without a pinned revision. To pin one, set
`QWEN_REVISION`, `MISTRAL_REVISION` or `YI_REVISION` for the run scripts.

## Data

GoEmotions and LeWiDi-MP are public datasets and are not redistributed. See `data/README.md` for where to
obtain the files, their checksums, and how to build `data/goe_test_prepared.json` and
`data/goe_reference_1200.npz`.

## Hardware and software

- 1x NVIDIA RTX PRO 6000 Blackwell (96 GB); one model on the GPU at a time.
- vLLM 0.30.0 (V1 engine), torch 2.14.0+cu130, transformers 5.17.0, CPython 3.10. The complete package set
  is in `requirements.txt`; `scripts/setup_env.sh` builds it. On our Blackwell (sm_120) system flashinfer
  needed `scripts/flashinfer_sm120.patch` (`APPLY_FLASHINFER_PATCH=1`).
- Analysis: CPython 3.11 with numpy, scipy and matplotlib (`requirements-analysis.txt`); no GPU needed.

## Reproducing the results

1. Environment and data

   ```bash
   VENV=$HOME/venvs/agg bash scripts/setup_env.sh          # add APPLY_FLASHINFER_PATCH=1 on sm_120 GPUs
   # download the datasets as described in data/README.md, then:
   python3 data/prepare_goemotions.py --raw_dir data/goemotions --out data/goe_test_prepared.json
   ```

2. Generation (GPU)

   ```bash
   PY=$HOME/venvs/agg/bin/python3 HF_CLI=$HOME/venvs/agg/bin/hf bash scripts/run_main.sh
   PY=$HOME/venvs/agg/bin/python3 HF_CLI=$HOME/venvs/agg/bin/hf bash scripts/run_dose_response.sh
   ```

   `run_main.sh` runs all main conditions (three model families x seeds 43/44/45; the Qwen-only
   baselines and the 2x2 format x K factorial) into `results/p1/`; `run_dose_response.sh` runs debate and
   MoA with L = 1, 2, 4 rounds on GoEmotions x Qwen3-32B into `results/p3/`. Both skip outputs that already
   exist; `DRY_RUN=1 bash scripts/run_main.sh` lists the pending jobs. Directories can be changed with
   `DATA_DIR`, `REF_DIR`, `RESULTS_DIR`, `LOG_DIR` and `MODEL_DIR`. Single jobs can also be run directly,
   e.g. `python3 generation/goe_debate.py --model <dir> --data data/goe_test_prepared.json --sc_ref
   ref/C_sc_probs.npz --seed 43 --out results/p1/C_goe_qwen_debateF_k5_v2_s43.npz`.

   Sampling settings (vLLM methods): temperature 0.7, top-p 0.95, at most 256 new tokens per call on
   GoEmotions (64 for LeWiDi independent sampling, 200 for LeWiDi debate/MoA), `max_model_len` 4096,
   prefix caching off, eager mode, one request per sample with its own seed derived from (run seed, item,
   sample). The transformers-based methods generate at most 64 new tokens per clarification (temperature
   0.7, top-p 0.95) or per CCD context (greedy) and read p(Yes) from next-token log-probabilities.

3. Analysis (CPU)

   ```bash
   bash scripts/run_analysis.sh
   ```

   This builds the reference file and the human reliability ceiling if they are missing, then runs
   `final_analysis.py`, `make_figures.py` and `make_prompts_appendix.py`; everything is written to
   `output/` (`numbers.tex`, `final_numbers.json`, `tab_ci_full.tex`, `tab_softce_sensitivity.tex`,
   `MISSING.md`, `figures/`, `appendix_prompts_body.tex`). `final_analysis.py` reads `RESULTS_DIR`
   (default `results/`), `GOE_REF`, `CEILING_JSON` and `OUT_DIR` from the environment; conditions whose
   files are absent are listed in `MISSING.md`. The bootstrap (B = 5000) takes a few minutes.

## Chat templates and BOS

All prompts are single user turns rendered with the model's own chat template
(`apply_chat_template(add_generation_prompt=True)`; for Qwen3 with `enable_thinking=False`); no system
prompt is added. Two details matter for Mistral and Yi with vLLM 0.30.0 / transformers 5.17.0:

- BOS. The Mistral template renders `<s>` as text, and vLLM adds BOS itself when it tokenizes a text
  prompt. The scripts therefore remove a leading BOS string from the rendered prompt, so that each prompt
  starts with exactly one BOS.
- Chat markers. For Yi-1.5-34B-Chat the vLLM tokenizer does not map `<|im_start|>` / `<|im_end|>` to their
  single ids, and for a complete Mistral directory the engine's mistral_common tokenizer treats
  `[INST]` / `[/INST]` in a text prompt as plain text. `prompt_encoding.py` detects this once per run and
  then passes token ids encoded with the model's `tokenizer.json`, prepending BOS exactly when vLLM would
  (Mistral `[1]`, Yi none). Each run logs `[prompt-encoding] ...` lines; `run_main.sh` checks them and
  stops on a mismatch. `generation/checks/check_mistral_tokens.py` verifies that the Mistral ids equal the
  official mistral_common chat encoding.

Qwen3 prompts are passed as text.

## Nondeterminism

Every sample has its own seed and prefix caching is disabled, so the sampled outputs do not depend on
which other requests share a cache. Batched GPU inference is nevertheless not bitwise deterministic:
batch composition, kernels, hardware, driver and library versions can change floating-point results and
hence sampled tokens. A rerun on another system can therefore give slightly different `.npz` files. The
numbers in the paper are computed from the files in `results/`; `final_analysis.py` is deterministic
given those files (bootstrap seed 20260929).

## Citation

```bibtex
@article{aggregation_attenuates,
  title   = {Aggregation Attenuates the Human Disagreement Signal in Compute-Matched Multi-Agent LLM Soft-Label Prediction},
  author  = {Keito Inoshita},
  journal = {Neurocomputing},
  note    = {Under review},
  year    = {2026}
}
```

## License

MIT; see `LICENSE`. The datasets and model weights are subject to their own licenses.
