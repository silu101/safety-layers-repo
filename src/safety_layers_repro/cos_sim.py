"""
Core logic for the N-N / M-M / N-M layer-wise cosine similarity analysis
(paper Section 3.2-3.3; original script:
Code/Cos_sim_analysis/save_all_pairs_cos_sim.py).

Design note: the pure numeric pieces (`cosine_similarity`,
`layerwise_cosine_similarity`) take plain vectors/arrays and have no model
or randomness dependency, so they're unit-testable without a GPU or model
weights (see tests/test_cos_sim.py). The model-dependent pieces
(`get_last_position_hidden_states`, `sample_query_pair`,
`run_pair_type`) are kept thin wrappers around those pure functions.
"""
from __future__ import annotations

import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import CosSimConfig
from .prompter import Prompter


# --------------------------------------------------------------------------
# Pure numeric helpers (no model / no I/O -> unit-testable without a GPU)
# --------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors. Matches the original
    script's inline computation exactly (np.dot / (norm * norm))."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def layerwise_cosine_similarity(
    vectors_a: Sequence[np.ndarray], vectors_b: Sequence[np.ndarray]
) -> List[float]:
    """Given two equal-length lists of per-layer last-position vectors
    (one list per query in the pair), return the list of cosine similarities
    layer-by-layer. This is L_C in the paper's notation."""
    if len(vectors_a) != len(vectors_b):
        raise ValueError(
            f"Layer count mismatch: {len(vectors_a)} vs {len(vectors_b)}. "
            "This happens if the two queries were tokenized/generated with "
            "different hidden_states structure -- check the model config."
        )
    return [cosine_similarity(va, vb) for va, vb in zip(vectors_a, vectors_b)]


def load_sentences(csv_path: str) -> List[str]:
    """Load a single-column, headerless CSV of instructions, matching the
    original script's `pd.read_csv(path, header=None)` + `df[0].tolist()`."""
    df = pd.read_csv(csv_path, header=None)
    return df[0].tolist()


def sample_query_pair(
    sentences_a: Sequence[str],
    sentences_b: Sequence[str],
    same_pool: bool,
    seed: int,
) -> Tuple[str, str]:
    """Draw one (instruction1, instruction2) pair for a given seed, matching
    the original script's branching: if the two CSV paths are identical
    (same_pool=True), sample 2 distinct items from one pool; otherwise sample
    1 item from each pool independently."""
    random.seed(seed)
    if same_pool:
        item1, item2 = random.sample(list(sentences_a), 2)
        return item1, item2
    item1 = random.sample(list(sentences_a), 1)[0]
    item2 = random.sample(list(sentences_b), 1)[0]
    return item1, item2


# --------------------------------------------------------------------------
# Model-dependent pieces
# --------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: CosSimConfig):
    """Deferred heavy import so that pure-logic tests don't require torch/
    transformers to be installed at all."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(cfg.dtype, "auto")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_path,
        trust_remote_code=cfg.trust_remote_code,
        padding_side="right",
        use_fast=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        device_map=cfg.device_map,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tokenizer


def get_last_position_hidden_states(
    model, tokenizer, prompter: Prompter, instruction: str, cfg: CosSimConfig
) -> List[np.ndarray]:
    """Run one forward/generate pass and extract the last-token hidden state
    from every hidden layer (skipping the embedding layer, index 0 -- this
    matches the original script's `for i in range(len(hs1[0])): if i==0:
    continue`). Returns one np.ndarray per layer."""
    import torch
    from transformers import GenerationConfig

    prompt = prompter.generate_prompt(instruction, None)
    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    # NOTE: num_beams is intentionally NOT passed here, even though cfg.num_beams
    # exists and defaults to 4. The original script (save_all_pairs_cos_sim.py)
    # takes num_beams as a get_output() parameter but never actually forwards it
    # into GenerationConfig or generate() -- a bug in the authors' own code that
    # makes their real executed behavior plain greedy decoding (num_beams=1,
    # do_sample defaults to False so temperature/top_p/top_k are inert too),
    # not the beam search the parameter name implies. See
    # docs/KNOWN_DISCREPANCIES.md #10. We match their ACTUAL behavior, not their
    # apparent intent, since that's what generated the paper's real numbers.
    generation_config = GenerationConfig(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        top_k=cfg.top_k,
        pad_token_id=cfg.pad_token_id,
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            output_hidden_states=True,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=cfg.max_new_tokens,
            num_return_sequences=1,
        )
    hidden_states = out["hidden_states"]  # tuple over generation steps
    step0_layers = hidden_states[0]  # first generation step: tuple over layers (incl. embedding)
    vectors = []
    for i in range(len(step0_layers)):
        if i == 0:
            continue  # skip embedding layer, matches original script
        # .numpy() doesn't support bfloat16 directly (e.g. under dtype=auto,
        # which resolves to this model's default torch_dtype); cast to
        # float32 first. No-op for configs already running in float32.
        vectors.append(step0_layers[i][0][-1].detach().to(torch.float32).cpu().numpy())
    return vectors


def run_pair_type(
    model,
    tokenizer,
    prompter: Prompter,
    path_a: str,
    path_b: str,
    seed: int,
    r: int,
    cfg: CosSimConfig,
    progress: bool = True,
) -> List[List[float]]:
    """Reproduces get_r_lists_cossim from the original script: draws r query
    pairs and returns r lists of per-layer cosine similarities."""
    sentences_a = load_sentences(path_a)
    sentences_b = load_sentences(path_b)
    same_pool = path_a == path_b

    iterator = range(r)
    if progress:
        from tqdm import tqdm

        iterator = tqdm(iterator)

    all_cos: List[List[float]] = []
    current_seed = seed
    for _ in iterator:
        inst_a, inst_b = sample_query_pair(sentences_a, sentences_b, same_pool, current_seed)
        current_seed += 1  # matches original script's `seed = seed + 1` per draw

        vectors_a = get_last_position_hidden_states(model, tokenizer, prompter, inst_a, cfg)
        vectors_b = get_last_position_hidden_states(model, tokenizer, prompter, inst_b, cfg)
        all_cos.append(layerwise_cosine_similarity(vectors_a, vectors_b))
    return all_cos
