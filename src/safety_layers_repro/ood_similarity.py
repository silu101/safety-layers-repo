"""
Embedding-similarity check for semantic-content OOD candidate datasets,
per the design discussion in this project: a candidate dataset's category
only counts as genuinely OOD relative to an ID anchor if its prompts are
*embedding-distant* from the ID set, not just differently named (the
AdvBench/HarmBench "different name, same content" problem).

ID anchor is AdvBench (520 prompts, data/advbench_malicious.csv) -- NOT
HarmBench. HarmBench is used elsewhere in this project only as an
independent judge/classifier (harmbench_classifier.py) for scoring
responses; it was never the prompt set our fine-tuned models were
actually evaluated against. AdvBench is the fixed evaluation distribution
our whole Section 4 pipeline anchors on, so it's the correct reference
for "did this generalize beyond what it was tested on."

Uses a local sentence-embedding model (no API cost, no rate limits) --
`all-mpnet-base-v2` by default, matching the model already validated in
this project's own exploratory check.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "all-mpnet-base-v2"


def load_embedder(model_name: str = DEFAULT_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed(model, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)


def nearest_neighbor_similarity(candidate_emb: np.ndarray, id_emb: np.ndarray) -> np.ndarray:
    """For each candidate embedding, its cosine similarity to its single
    nearest neighbor in the ID embedding set. Both must be normalized
    (embed() does this), so a plain dot product IS cosine similarity."""
    sims = candidate_emb @ id_emb.T
    return sims.max(axis=1)


def nearest_neighbor_index_and_similarity(candidate_emb: np.ndarray, id_emb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Like nearest_neighbor_similarity, but also returns WHICH id_emb row
    was the nearest neighbor for each candidate -- needed to actually show
    a human the matched (candidate, id) prompt pair, not just the score."""
    sim_matrix = candidate_emb @ id_emb.T
    idx = sim_matrix.argmax(axis=1)
    sims = sim_matrix[np.arange(len(candidate_emb)), idx]
    return idx, sims


def per_category_summary(
    categories: Sequence[str], similarities: np.ndarray, threshold: float = 0.7,
) -> Dict[str, dict]:
    """Groups per-prompt nearest-neighbor similarities by category and
    reports summary stats + how many would survive a given similarity
    threshold (below threshold = OOD candidate, kept)."""
    cats = np.array(categories)
    summary = {}
    for cat in sorted(set(categories)):
        mask = cats == cat
        sims = similarities[mask]
        survivors = int((sims < threshold).sum())
        summary[cat] = {
            "n": int(mask.sum()),
            "mean_similarity": float(sims.mean()),
            "median_similarity": float(np.median(sims)),
            "min_similarity": float(sims.min()),
            "max_similarity": float(sims.max()),
            "n_survive_threshold": survivors,
            "pct_survive_threshold": survivors / int(mask.sum()) if mask.sum() else 0.0,
        }
    return summary


def run_similarity_check(
    id_prompts: Sequence[str],
    candidate_prompts: Sequence[str],
    candidate_categories: Sequence[str],
    threshold: float = 0.7,
    model_name: str = DEFAULT_MODEL,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    """Full pipeline: embed ID + candidates, compute nearest-neighbor
    similarity per candidate, summarize per category. Returns
    (per-prompt similarity array, per-category summary dict)."""
    model = load_embedder(model_name)
    id_emb = embed(model, id_prompts)
    cand_emb = embed(model, candidate_prompts)
    sims = nearest_neighbor_similarity(cand_emb, id_emb)
    summary = per_category_summary(candidate_categories, sims, threshold=threshold)
    return sims, summary


def sample_nearest_pairs(
    candidate_prompts: Sequence[str],
    candidate_categories: Sequence[str],
    id_prompts: Sequence[str],
    category: Optional[str] = None,
    n: int = 20,
    mode: str = "random",
    model_name: str = DEFAULT_MODEL,
    seed: int = 0,
) -> List[dict]:
    """Manual-verification helper: for a spot-check, pulls candidate prompts
    together with the SPECIFIC AdvBench prompt each one matched and its
    similarity score, so a human can read the matched pair side by side
    rather than judge a category off its mean alone.

    category: restrict to one candidate_category (e.g. "BeaverTails::self_harm").
        If None, samples across all categories given.
    mode:
        "random"  -- uniform random sample of size n (default; use for a
                     general "does this category look sane" check).
        "highest" -- the n candidates with the HIGHEST similarity to their
                     nearest AdvBench neighbor (use to check whether a
                     wide-spread category is hiding near-duplicates of
                     AdvBench at its high end).
        "lowest"  -- the n candidates with the LOWEST similarity (use to
                     check whether low-similarity prompts are genuinely
                     off-topic/mislabeled rather than just OOD).

    Returns a list of dicts, each: {candidate_prompt, candidate_category,
    similarity, matched_id_prompt}, sorted by similarity descending.
    """
    cats = np.array(candidate_categories)
    if category is not None:
        mask = cats == category
        prompts = [p for p, m in zip(candidate_prompts, mask) if m]
        cats_kept = cats[mask]
    else:
        prompts = list(candidate_prompts)
        cats_kept = cats

    if not prompts:
        return []

    model = load_embedder(model_name)
    cand_emb = embed(model, prompts)
    id_emb = embed(model, id_prompts)
    nn_idx, sims = nearest_neighbor_index_and_similarity(cand_emb, id_emb)

    order = np.argsort(-sims)  # descending similarity
    if mode == "highest":
        chosen = order[:n]
    elif mode == "lowest":
        chosen = order[::-1][:n]
    elif mode == "random":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(prompts), size=min(n, len(prompts)), replace=False)
        chosen = chosen[np.argsort(-sims[chosen])]
    else:
        raise ValueError(f"unknown mode: {mode!r} (expected random/highest/lowest)")

    return [
        {
            "candidate_prompt": prompts[i],
            "candidate_category": str(cats_kept[i]),
            "similarity": float(sims[i]),
            "matched_id_prompt": id_prompts[nn_idx[i]],
        }
        for i in chosen
    ]


def dedup_within_pool(
    prompts: Sequence[str], embeddings: np.ndarray, threshold: float = 0.9,
) -> List[int]:
    """Greedy near-duplicate removal within a single merged pool (multiple
    source datasets scraped from overlapping places, e.g. templated
    BeaverTails/Aegis prompts). Returns the indices to KEEP, in original
    order. O(n^2) pairwise cosine sim -- fine up to a few tens of thousands
    of prompts (embeddings are already normalized, so dot product = cosine)."""
    n = len(prompts)
    if n == 0:
        return []
    sim_matrix = embeddings @ embeddings.T
    keep: List[int] = []
    kept_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if kept_mask.any() and sim_matrix[i, kept_mask].max() >= threshold:
            continue
        keep.append(i)
        kept_mask[i] = True
    return keep


def build_ood_pool(
    category_map: Dict[str, List[Tuple[str, str]]],
    source_prompts: Dict[str, Dict[str, List[str]]],
    id_prompts: Optional[Sequence[str]] = None,
    dedup_threshold: float = 0.9,
    max_similarity: Optional[float] = None,
    model_name: str = DEFAULT_MODEL,
    verbose: bool = False,
) -> Dict[str, dict]:
    """Builds provenance-tagged, deduplicated, per-prompt-filtered OOD
    prompt pools per merged category -- the generic, method-agnostic step:
    output is just (prompt, source_dataset, source_label, merged_category)
    records, usable by any of the 8 methods' OOD evaluation, not specific
    to Safety Layers.

    category_map: {merged_category: [(source_dataset_name, source_label), ...]}
    source_prompts: {source_dataset_name: {source_label: [prompts...]}}
    max_similarity: if given (requires id_prompts), drops any individual
        prompt whose nearest-neighbor similarity to id_prompts is >= this
        value, even from a category whose overall mean is well below it.
        Intended to be set to a positive-control baseline (e.g. the
        AdvBench<->HarmBench known-overlap mean) -- a category can look
        confidently OOD on average while still containing a handful of
        prompts that are themselves indistinguishable from known overlap.
        (i.e. the raw (prompt, category) pairs each dataset loader already
        produces, regrouped by label)
    id_prompts: if given, re-runs the AdvBench similarity check on each
        deduplicated merged pool as a sanity re-check post-merge.

    Returns {merged_category: {"records": [...], "n_before_dedup": int,
             "n_after_dedup": int, "similarity_recheck": {...} | None}}
    """
    if verbose:
        print("[build_ood_pool] loading embedder...", flush=True)
    model = load_embedder(model_name)
    if verbose:
        print("[build_ood_pool] embedder loaded", flush=True)
    id_emb = embed(model, id_prompts) if id_prompts is not None else None
    if verbose and id_emb is not None:
        print(f"[build_ood_pool] embedded {len(id_prompts)} ID prompts", flush=True)

    result: Dict[str, dict] = {}
    for merged_cat, sources in category_map.items():
        records = []
        for source_name, source_label in sources:
            for prompt in source_prompts.get(source_name, {}).get(source_label, []):
                records.append({
                    "prompt": prompt,
                    "source_dataset": source_name,
                    "source_label": source_label,
                    "merged_category": merged_cat,
                })
        n_before = len(records)
        if verbose:
            print(f"[build_ood_pool] {merged_cat}: {n_before} prompts before dedup, embedding...", flush=True)

        if n_before == 0:
            result[merged_cat] = {
                "records": [], "n_before_dedup": 0, "n_after_dedup": 0, "similarity_recheck": None,
            }
            continue

        pool_prompts = [r["prompt"] for r in records]
        pool_emb = embed(model, pool_prompts)
        if verbose:
            print(f"[build_ood_pool] {merged_cat}: embedded, running dedup (threshold={dedup_threshold})...", flush=True)
        keep_idx = dedup_within_pool(pool_prompts, pool_emb, threshold=dedup_threshold)
        deduped_records = [records[i] for i in keep_idx]
        n_after = len(deduped_records)
        if verbose:
            print(f"[build_ood_pool] {merged_cat}: {n_before} -> {n_after} after dedup", flush=True)

        recheck = None
        n_filtered_out = 0
        if id_emb is not None and n_after > 0:
            deduped_emb = pool_emb[keep_idx]
            nn_idx, sims = nearest_neighbor_index_and_similarity(deduped_emb, id_emb)
            for rec, sim, idx in zip(deduped_records, sims, nn_idx):
                rec["advbench_similarity"] = float(sim)
                rec["matched_advbench_prompt"] = id_prompts[idx]

            # Per-prompt filter: a category can average well below the known-
            # overlap baseline while still containing individual prompts that
            # are themselves at or above it -- the category-level mean never
            # catches that. Drop those regardless of the category's overall
            # verdict; max_similarity is meant to be set to the positive-
            # control baseline (e.g. AdvBench<->HarmBench mean) by the caller.
            if max_similarity is not None:
                keep_mask = sims < max_similarity
                n_filtered_out = int((~keep_mask).sum())
                deduped_records = [r for r, keep in zip(deduped_records, keep_mask) if keep]
                sims = sims[keep_mask]
                n_after = len(deduped_records)
                if verbose and n_filtered_out:
                    print(f"[build_ood_pool] {merged_cat}: dropped {n_filtered_out} prompts "
                          f"at/above max_similarity={max_similarity:.3f}", flush=True)

            if len(sims) > 0:
                recheck = {
                    "mean_similarity": float(sims.mean()),
                    "median_similarity": float(np.median(sims)),
                    "pct_survive_0.7": float((sims < 0.7).mean()),
                }
                if verbose:
                    print(f"[build_ood_pool] {merged_cat}: post-filter mean sim to AdvBench = {recheck['mean_similarity']:.3f}", flush=True)

        result[merged_cat] = {
            "records": deduped_records,
            "n_before_dedup": n_before,
            "n_after_dedup": n_after,
            "n_filtered_by_similarity": n_filtered_out,
            "similarity_recheck": recheck,
        }
    return result
