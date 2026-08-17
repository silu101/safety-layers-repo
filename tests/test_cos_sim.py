"""
Tests for the pure (non-model) logic in safety_layers_repro.cos_sim.

These run on CPU with no GPU, no model download, and no torch/transformers
dependency, so they can run in CI on every PR as a fast correctness check
of the sampling and math logic, independent of any model-specific run.
"""
import numpy as np
import pytest

from safety_layers_repro.cos_sim import (
    cosine_similarity,
    layerwise_cosine_similarity,
    load_sentences,
    sample_query_pair,
)


def test_cosine_similarity_identical_vectors():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    a = np.array([1.0, 0.0])
    b = np.array([-1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_scale_invariant():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([2.0, 4.0, 6.0])
    assert cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_zero_vector_returns_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert np.isnan(cosine_similarity(a, b))


def test_layerwise_cosine_similarity_basic():
    vectors_a = [np.array([1.0, 0.0]), np.array([1.0, 1.0])]
    vectors_b = [np.array([1.0, 0.0]), np.array([1.0, -1.0])]
    result = layerwise_cosine_similarity(vectors_a, vectors_b)
    assert len(result) == 2
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(0.0)


def test_layerwise_cosine_similarity_length_mismatch_raises():
    vectors_a = [np.array([1.0, 0.0])]
    vectors_b = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    with pytest.raises(ValueError):
        layerwise_cosine_similarity(vectors_a, vectors_b)


def test_load_sentences(tmp_path):
    csv_path = tmp_path / "sentences.csv"
    csv_path.write_text("How to kill time?\nHow to bake bread?\n")
    sentences = load_sentences(str(csv_path))
    assert sentences == ["How to kill time?", "How to bake bread?"]


def test_sample_query_pair_same_pool_returns_two_distinct_items():
    sentences = ["a", "b", "c", "d"]
    item1, item2 = sample_query_pair(sentences, sentences, same_pool=True, seed=0)
    assert item1 != item2
    assert item1 in sentences and item2 in sentences


def test_sample_query_pair_different_pools():
    normal = ["n1", "n2"]
    malicious = ["m1", "m2"]
    item1, item2 = sample_query_pair(normal, malicious, same_pool=False, seed=0)
    assert item1 in normal
    assert item2 in malicious


def test_sample_query_pair_is_deterministic_given_seed():
    sentences = ["a", "b", "c", "d", "e"]
    pair1 = sample_query_pair(sentences, sentences, same_pool=True, seed=42)
    pair2 = sample_query_pair(sentences, sentences, same_pool=True, seed=42)
    assert pair1 == pair2


def test_sample_query_pair_different_seeds_can_differ():
    # Not a strict guarantee for every seed pair, but true for this fixture --
    # documents the seed's role rather than testing randomness quality itself.
    sentences = [f"item_{i}" for i in range(50)]
    pair1 = sample_query_pair(sentences, sentences, same_pool=True, seed=1)
    pair2 = sample_query_pair(sentences, sentences, same_pool=True, seed=2)
    assert pair1 != pair2
