"""
Tests for safety_layers_repro.compare using fully synthetic all_cos-style
data. These validate the comparison/report logic itself -- independent of
whether any real model run has happened -- so a teammate can trust the
comparison tool before spending GPU time on a real reproduction.
"""
import pickle

import numpy as np
import pytest

from safety_layers_repro.compare import (
    compare_all,
    compare_pair_type,
    find_gap_onset,
    load_all_cos,
)


def make_synthetic_all_cos(n_draws: int, n_layers: int, seed: int, gap_layer: int = None):
    """Build a synthetic [NN, MM, NM] result. If gap_layer is given, the N-M
    curve is constructed to drop noticeably starting at that layer, mimicking
    the paper's claimed pattern; NN and MM stay high throughout."""
    rng = np.random.default_rng(seed)

    def curve(base_high=True, drop_at=None):
        vals = []
        for layer in range(n_layers):
            if drop_at is not None and layer >= drop_at:
                mean = 0.5
            else:
                mean = 0.95 if base_high else 0.9
            vals.append(mean)
        return np.array(vals)

    def draws(mean_curve):
        return [
            (mean_curve + rng.normal(0, 0.01, size=n_layers)).tolist()
            for _ in range(n_draws)
        ]

    nn = draws(curve(base_high=True))
    mm = draws(curve(base_high=True))
    nm = draws(curve(base_high=True, drop_at=gap_layer))
    return [nn, mm, nm]


def test_load_all_cos_roundtrip(tmp_path):
    data = make_synthetic_all_cos(n_draws=5, n_layers=8, seed=0, gap_layer=4)
    path = tmp_path / "synthetic.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)
    loaded = load_all_cos(str(path))
    assert len(loaded) == 3
    assert len(loaded[0]) == 5


def test_load_all_cos_rejects_wrong_shape(tmp_path):
    path = tmp_path / "bad.pkl"
    with open(path, "wb") as f:
        pickle.dump([1, 2], f)  # not a 3-element list
    with pytest.raises(ValueError):
        load_all_cos(str(path))


def test_find_gap_onset_detects_the_configured_layer():
    n_layers = 10
    mean_nn = np.full(n_layers, 0.95)
    mean_nm = np.array([0.95] * 4 + [0.5] * 6)
    onset = find_gap_onset(mean_nn, mean_nm, margin=0.05)
    assert onset == 4


def test_find_gap_onset_returns_none_when_no_gap():
    n_layers = 10
    mean_nn = np.full(n_layers, 0.95)
    mean_nm = np.full(n_layers, 0.93)
    onset = find_gap_onset(mean_nn, mean_nm, margin=0.05)
    assert onset is None


def test_compare_pair_type_identical_data_gives_perfect_agreement():
    data = make_synthetic_all_cos(n_draws=20, n_layers=6, seed=1, gap_layer=3)
    result = compare_pair_type(data[2], data[2], "N-M")
    assert result.mean_abs_diff == pytest.approx(0.0, abs=1e-9)
    assert result.pearson_r == pytest.approx(1.0, abs=1e-6)


def test_compare_all_recovers_same_gap_onset_for_identical_files(tmp_path):
    data = make_synthetic_all_cos(n_draws=50, n_layers=12, seed=2, gap_layer=5)
    path_a = tmp_path / "a.pkl"
    path_b = tmp_path / "b.pkl"
    with open(path_a, "wb") as f:
        pickle.dump(data, f)
    with open(path_b, "wb") as f:
        pickle.dump(data, f)

    report = compare_all(str(path_a), str(path_b))
    onset = report["safety_layer_gap_onset"]
    assert onset["ours"] == onset["reference"]
    assert onset["ours"] == 5


def test_compare_all_flags_small_sample_warning(tmp_path):
    data = make_synthetic_all_cos(n_draws=5, n_layers=8, seed=3, gap_layer=4)
    path = tmp_path / "small.pkl"
    with open(path, "wb") as f:
        pickle.dump(data, f)

    report = compare_all(str(path), str(path))
    assert any("draws" in w for w in report["warnings"])


def test_compare_all_handles_mismatched_layer_counts(tmp_path):
    data_8 = make_synthetic_all_cos(n_draws=20, n_layers=8, seed=4, gap_layer=4)
    data_10 = make_synthetic_all_cos(n_draws=20, n_layers=10, seed=4, gap_layer=4)
    path_a = tmp_path / "a.pkl"
    path_b = tmp_path / "b.pkl"
    with open(path_a, "wb") as f:
        pickle.dump(data_8, f)
    with open(path_b, "wb") as f:
        pickle.dump(data_10, f)

    with pytest.warns(UserWarning, match="Layer count mismatch"):
        report = compare_all(str(path_a), str(path_b))
    # Should still produce a report rather than crashing.
    assert "N-N" in report["per_pair_type"]
