# Results

This directory holds the generated model outputs (`.npz`) analysed in the paper.

Layout expected by the analysis scripts:

```
results/
  p1/   main conditions          C_goe_<model>_<method>_v2_s<seed>.npz   (GoEmotions, 1,200 items x 28 labels)
                                 C_mp_<model>_<method>_v2_s<seed>.npz    (LeWiDi-MP, 3,756 items x 2)
  p3/   dose-response (Qwen3-32B) C_goe_qwen_{debateF,moaF}_L<L>_k5_v2_s<seed>.npz, L in {1, 2, 4}
```

`<model>` is `qwen`, `mistral` or `yi`; seeds are 43, 44 and 45. Each file stores the predicted soft
labels (`p`, or `p_hou` / `p_ccd` and `p_base` for the transformers-based methods) and the item `ids`;
debate, MoA and LeWiDi files also store the human soft labels `p_human`.

Generation is not bitwise reproducible on every system (see the main README), so the analysis of the
paper should be reproduced from these files.

## Attribution of human soft labels

The files store item identifiers and probability vectors only; they contain no text. The human soft labels
(`p_human`) are aggregated from the public annotations of GoEmotions (Demszky et al., 2020) and of MultiPICo
(Casola et al., 2024) as distributed in the LeWiDi-2025 shared task (Leonardelli et al., 2025). Please cite
these datasets and follow their licenses when reusing these labels.
