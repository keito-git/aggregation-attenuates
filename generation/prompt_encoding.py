"""Chat-marker-safe prompt encoding for vLLM text prompts.

Under transformers 5.17 some tokenizers (Yi-1.5-34B-Chat) do not register chat markers such as
<|im_start|> as special tokens, because the marker is listed in tokenizer.json but not in
tokenizer_config.json. vLLM then splits the marker into ordinary characters, so the model never
sees its chat format. Likewise, for a complete Mistral-Small-24B-Instruct-2501 directory the vLLM
engine uses the mistral_common tokenizer, which treats [INST] / [/INST] in a text prompt as plain text.
This module checks, once per run, whether vLLM's tokenizer maps every special marker that occurs in
the rendered prompts to its single id. If it does (Qwen), prompts are passed as text. If not (Yi,
Mistral), prompts are encoded with the raw tokenizers.Tokenizer from tokenizer.json (which registers
the markers) and passed to vLLM as token ids, with the BOS id prepended exactly when vLLM would prepend
it to a text prompt (Mistral: [1]; Yi: none).
"""
import json
import os


def _special_added_tokens(model_dir):
    path = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        tj = json.load(f)
    return {a["content"]: a["id"] for a in tj.get("added_tokens", []) if a.get("special")}


def make_prompt_adapter(tokenizer, model_dir, sample_prompts):
    """Return f(list[str]) -> list of vLLM prompts (texts, or {'prompt_token_ids': ids})."""
    specials = _special_added_tokens(model_dir)
    used = {m: i for m, i in specials.items() if any(m in s for s in sample_prompts)}
    broken = {m: i for m, i in used.items()
              if list(tokenizer.encode(m, add_special_tokens=False)) != [i]}
    if not broken:
        print(f"[prompt-encoding] vLLM tokenizer maps chat markers {sorted(used)} to single ids; "
              f"passing text prompts", flush=True)
        return lambda texts: texts

    from tokenizers import Tokenizer
    raw = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    for m, i in used.items():
        got = raw.encode(m, add_special_tokens=False).ids
        if got != [i]:
            raise RuntimeError(f"tokenizer.json also fails to encode {m!r} as [{i}] (got {got})")
    print(f"[prompt-encoding] vLLM tokenizer splits chat markers {sorted(broken)}; "
          f"passing token ids encoded with tokenizer.json", flush=True)

    # BOS: mirror what vLLM itself does for a text prompt. If vLLM prepends its BOS id to a text prompt
    # (Mistral), prepend it here too; if it does not (Yi, Qwen), add nothing.
    probe = list(tokenizer.encode("probe"))
    bos_id = getattr(tokenizer, "bos_token_id", None)
    prefix = [bos_id] if (bos_id is not None and probe and probe[0] == bos_id) else []
    print(f"[prompt-encoding] BOS prefix on the token-id path: {prefix}", flush=True)

    def adapt(texts):
        return [{"prompt_token_ids": prefix + raw.encode(t, add_special_tokens=False).ids} for t in texts]
    return adapt
