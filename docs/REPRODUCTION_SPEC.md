# Reproduction Spec: Cosine Similarity Existence Check (Paper Section 3.2–3.3)

Extracted directly from `Code/Cos_sim_analysis/save_all_pairs_cos_sim.py` in the
[original repo](https://github.com/listen0425/Safety-Layers) (commit at time of
writing: check `reference/` provenance / clone date below), not from the paper
text alone, per the project's rule to prefer authors' code over reimplementing
from prose.

## Input → Method → Parameters → Evaluation → Claimed Result

| Stage | Detail |
|---|---|
| **Input** | `data/normal.csv` (99 rows) and `data/malicious.csv` (100 rows). Single-column, no header. Loaded via `pd.read_csv(path, header=None)[0].tolist()`. |
| **Model** | Any of the paper's 4 aligned LLMs. This repo's default config targets `microsoft/Phi-3-mini-4k-instruct` (ungated, cheapest to start with). Others: `meta-llama/Llama-2-7b-chat-hf`, `meta-llama/Meta-Llama-3-8B-Instruct` (gated, need HF access approval), `google/gemma-2b-it`. |
| **Prompt template** | Alpaca, no-input variant (`src/safety_layers_repro/alpaca.json`): `"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"` |
| **Method** | For `r` random draws per pair-type: sample a query pair (see sampling rules below), run `model.generate(..., output_hidden_states=True, max_new_tokens=1)`, extract the **last-token hidden state at every layer except the embedding layer** (`hidden_states[0]` = first/only generation step, skip index 0), compute cosine similarity between the two queries' vectors layer-by-layer. |
| **Pair types & sampling** | **N-N**: 2 distinct items sampled from `normal.csv` (same pool → `random.sample(pool, 2)`). **M-M**: 2 distinct items from `malicious.csv`. **N-M**: 1 item each from `normal.csv` and `malicious.csv` independently. |
| **Seeds** | N-N: `seed=10`, incremented by 1 each draw. M-M: `seed=100`. N-M: `seed=1000`. (These specific values are arbitrary but must match exactly for a bitwise-comparable reproduction of *which* pairs were drawn — the paper's claim is about the aggregate statistic, not these specific pairs, so exact seed match is not required for a valid reproduction, only for an *identical* one.) |
| **r** | 500 (paper's setting; `configs/phi3_cossim_smoketest.yaml` uses 20 for a fast pipeline check only). |
| **Generation params** | `temperature=0.5, top_p=0.2, top_k=40, num_beams=4, max_new_tokens=1, pad_token_id=0`. Note: `num_beams=4` with `max_new_tokens=1` — beam search matters here because the hidden states extracted are from the beam search forward pass, not just greedy; confirm this reproduces if you switch decoding strategy. |
| **Output** | `all_cos.pkl`: `[NN_results, MM_results, NM_results]`, each a list of `r` lists, each inner list of length `K` (num hidden layers) of cosine similarities. |
| **Evaluation** | Compute per-layer mean and std across the `r` draws, for each pair-type (paper's Fig. 1/2 upper half). Plot as 3 side-by-side line charts with shaded std band. |
| **Claimed result** | (1) N-N and M-M mean curves stay high (~0.9+) and roughly flat across all layers. (2) N-M mean curve starts similarly high in early layers, then diverges downward starting at a specific middle layer, with the gap widening before leveling off in later layers. (3) This divergence point is claimed to mark the start of the "safety layers." No specific numeric threshold is given in the paper for what counts as "diverges" — Section 3.4 develops a separate, more precise localization method (over-rejection + parameter scaling) precisely because the cosine-similarity gap alone is described as imprecise for pinning down exact layer boundaries (paper, Section 3.4 opening paragraph). |

## What's precise vs. approximate in the paper's own words

The paper is explicit that this analysis only gives an **initial approximate range**, not precise boundaries (Section 3.4: *"locating the safety layers based solely on the range from the appearance of the gap to its increase until the first smoothing is imprecise"*). This repo's `compare.py` `find_gap_onset` heuristic should be read the same way — it's a rough, comparable summary statistic for checking two runs against each other, not a claim to have reproduced the paper's own localization algorithm (that's Section 3.4, not yet ported — see README "Scope").

## Known discrepancies between paper text and shipped code/data

See `docs/KNOWN_DISCREPANCIES.md`.
