"""
Numeric comparison of two `all_cos.pkl` results (e.g. your reproduction vs.
a reference/authors' file, or two teammates' runs of the same config).

This exists so reproduction can be judged by numbers, not by eyeballing
overlaid plots. It reports, per pair-type (N-N, M-M, N-M):
  - per-layer mean/std cosine similarity for each file
  - per-layer absolute difference of means
  - Pearson correlation between the two mean-curves across layers
  - the "gap onset" layer (see `find_gap_onset`) for each file, and its
    difference -- this is the core quantity behind the paper's claim
    ("a specific layer where N-M starts to diverge from N-N")

Usage:
    python -m safety_layers_repro.compare \\
        --ours results/<run>/all_cos.pkl \\
        --reference reference/all_cos_phi3_authors.pkl \\
        --out-dir results/<run>/comparison
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

PAIR_TYPE_NAMES = ["N-N", "M-M", "N-M"]


@dataclass
class PairTypeComparison:
    pair_type: str
    n_draws_ours: int
    n_draws_reference: int
    n_layers_ours: int
    n_layers_reference: int
    mean_curve_ours: List[float]
    mean_curve_reference: List[float]
    std_curve_ours: List[float]
    std_curve_reference: List[float]
    abs_diff_of_means: List[float]
    max_abs_diff: float
    mean_abs_diff: float
    pearson_r: Optional[float]
    gap_onset_layer_ours: Optional[int]
    gap_onset_layer_reference: Optional[int]


def _stack_ragged(list_of_lists: List[List[float]]) -> np.ndarray:
    """Convert r lists of K floats into an (r, K) array. Raises if the K
    (layer count) is inconsistent within a single file, since that would
    indicate a bug in the run rather than an expected difference."""
    lengths = {len(x) for x in list_of_lists}
    if len(lengths) > 1:
        raise ValueError(
            f"Inconsistent layer counts within one pair-type's draws: {lengths}. "
            "This should not happen for a single model/run; check the run for errors."
        )
    return np.asarray(list_of_lists, dtype=np.float64)


def find_gap_onset(
    mean_nn: np.ndarray, mean_nm: np.ndarray, margin: float = 0.05
) -> Optional[int]:
    """Find the first layer index where the N-N curve exceeds the N-M curve
    by more than `margin`, and the gap does not close again in the
    remaining layers' running minimum -- a simple, inspectable proxy for the
    "onset of the safety-layer gap" described qualitatively in the paper
    (Section 3.3). This is a heuristic, not the paper's own algorithm
    (Section 3.4 uses over-rejection scaling for precise localization) --
    use it only as a rough, comparable summary statistic between runs.
    """
    diff = mean_nn - mean_nm
    for i in range(len(diff)):
        if diff[i] > margin and np.all(diff[i:] > 0):
            return i
    return None


def compare_pair_type(
    ours: List[List[float]], reference: List[List[float]], pair_type: str
) -> PairTypeComparison:
    arr_ours = _stack_ragged(ours)
    arr_ref = _stack_ragged(reference)

    if arr_ours.shape[1] != arr_ref.shape[1]:
        warnings.warn(
            f"[{pair_type}] Layer count mismatch: ours={arr_ours.shape[1]} "
            f"vs reference={arr_ref.shape[1]}. Likely different models/architectures. "
            "Truncating to the shorter length for comparison; treat correlation/diff "
            "numbers here as informational only, not a pass/fail signal."
        )
    n_layers = min(arr_ours.shape[1], arr_ref.shape[1])

    mean_ours = arr_ours[:, :n_layers].mean(axis=0)
    std_ours = arr_ours[:, :n_layers].std(axis=0)
    mean_ref = arr_ref[:, :n_layers].mean(axis=0)
    std_ref = arr_ref[:, :n_layers].std(axis=0)

    abs_diff = np.abs(mean_ours - mean_ref)

    if n_layers >= 2 and np.std(mean_ours) > 0 and np.std(mean_ref) > 0:
        pearson_r = float(np.corrcoef(mean_ours, mean_ref)[0, 1])
    else:
        pearson_r = None

    onset_ours = find_gap_onset(mean_ours, mean_ours) if pair_type != "N-M" else None
    # Gap onset is only meaningful when comparing N-N vs N-M within the SAME
    # file; see compare_all() for the cross-pair-type computation. Here we
    # leave placeholders and fill them in from compare_all().

    return PairTypeComparison(
        pair_type=pair_type,
        n_draws_ours=arr_ours.shape[0],
        n_draws_reference=arr_ref.shape[0],
        n_layers_ours=arr_ours.shape[1],
        n_layers_reference=arr_ref.shape[1],
        mean_curve_ours=mean_ours.tolist(),
        mean_curve_reference=mean_ref.tolist(),
        std_curve_ours=std_ours.tolist(),
        std_curve_reference=std_ref.tolist(),
        abs_diff_of_means=abs_diff.tolist(),
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        pearson_r=pearson_r,
        gap_onset_layer_ours=None,
        gap_onset_layer_reference=None,
    )


def load_all_cos(path: str) -> List[List[List[float]]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not (isinstance(data, list) and len(data) == 3):
        raise ValueError(
            f"{path}: expected a list of 3 pair-type results ([NN, MM, NM]), "
            f"got a {type(data)} of length {len(data) if hasattr(data, '__len__') else 'unknown'}"
        )
    return data


def compare_all(ours_path: str, reference_path: str) -> dict:
    ours = load_all_cos(ours_path)
    reference = load_all_cos(reference_path)

    per_pair_type = {}
    for name, o, r in zip(PAIR_TYPE_NAMES, ours, reference):
        per_pair_type[name] = compare_pair_type(o, r, name)

    # Gap onset: compare N-N vs N-M mean curves, within each file separately.
    nn_ours = np.asarray(per_pair_type["N-N"].mean_curve_ours)
    nm_ours = np.asarray(per_pair_type["N-M"].mean_curve_ours)
    nn_ref = np.asarray(per_pair_type["N-N"].mean_curve_reference)
    nm_ref = np.asarray(per_pair_type["N-M"].mean_curve_reference)
    n_common = min(len(nn_ours), len(nm_ours))
    onset_ours = find_gap_onset(nn_ours[:n_common], nm_ours[:n_common])
    n_common_ref = min(len(nn_ref), len(nm_ref))
    onset_ref = find_gap_onset(nn_ref[:n_common_ref], nm_ref[:n_common_ref])
    per_pair_type["N-M"].gap_onset_layer_ours = onset_ours
    per_pair_type["N-M"].gap_onset_layer_reference = onset_ref

    small_sample_warning = None
    min_draws = min(
        per_pair_type[n].n_draws_ours for n in PAIR_TYPE_NAMES
    )
    if min_draws < 100:
        small_sample_warning = (
            f"At least one pair-type has only {min_draws} draws. The paper uses "
            "r=500; with small r, per-layer means/std are noisy and correlation "
            "numbers should not be treated as a strict pass/fail threshold. This "
            "is expected e.g. when comparing against the ~10-draw file shipped "
            "as a demo artifact in the original repo (reference/all_cos_phi3_authors.pkl)."
        )

    return {
        "per_pair_type": {k: asdict(v) for k, v in per_pair_type.items()},
        "safety_layer_gap_onset": {
            "ours": onset_ours,
            "reference": onset_ref,
            "note": (
                "Heuristic proxy (margin-based crossover of N-N vs N-M mean "
                "cosine similarity), not the paper's Section 3.4 localization "
                "algorithm. Use for quick comparison between runs only."
            ),
        },
        "warnings": [w for w in [small_sample_warning] if w],
    }


def render_markdown_report(report: dict, ours_path: str, reference_path: str) -> str:
    lines = [
        "# Cosine Similarity Reproduction Comparison",
        "",
        f"- Ours: `{ours_path}`",
        f"- Reference: `{reference_path}`",
        "",
    ]
    if report["warnings"]:
        lines.append("## Warnings")
        for w in report["warnings"]:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.append("## Per pair-type summary")
    lines.append("")
    lines.append("| Pair type | Draws (ours/ref) | Layers (ours/ref) | Mean |Δmean| | Max |Δmean| | Pearson r (mean curves) |")
    lines.append("|---|---|---|---|---|---|")
    for name in PAIR_TYPE_NAMES:
        p = report["per_pair_type"][name]
        pr = "n/a" if p["pearson_r"] is None else f"{p['pearson_r']:.4f}"
        lines.append(
            f"| {name} | {p['n_draws_ours']}/{p['n_draws_reference']} "
            f"| {p['n_layers_ours']}/{p['n_layers_reference']} "
            f"| {p['mean_abs_diff']:.4f} | {p['max_abs_diff']:.4f} | {pr} |"
        )
    lines.append("")

    onset = report["safety_layer_gap_onset"]
    lines.append("## Safety-layer gap onset (heuristic)")
    lines.append("")
    lines.append(f"- Ours: layer {onset['ours']}")
    lines.append(f"- Reference: layer {onset['reference']}")
    lines.append(f"- Note: {onset['note']}")
    lines.append("")
    return "\n".join(lines)


def maybe_plot(report: dict, out_dir: Path) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, name in zip(axes, PAIR_TYPE_NAMES):
        p = report["per_pair_type"][name]
        ax.plot(p["mean_curve_ours"], label="ours")
        ax.plot(p["mean_curve_reference"], label="reference", linestyle="--")
        ax.set_title(name)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean cos sim")
        ax.legend()
    fig.tight_layout()
    out_path = out_dir / "comparison_plot.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, help="Path to your all_cos.pkl")
    parser.add_argument("--reference", required=True, help="Path to the reference all_cos.pkl")
    parser.add_argument("--out-dir", default=None, help="Where to write the report (default: alongside --ours)")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.ours).parent / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    report = compare_all(args.ours, args.reference)

    with open(out_dir / "comparison_report.json", "w") as f:
        json.dump(report, f, indent=2)

    md = render_markdown_report(report, args.ours, args.reference)
    with open(out_dir / "comparison_report.md", "w") as f:
        f.write(md)

    plot_path = maybe_plot(report, out_dir)

    print(md)
    print(f"\nSaved: {out_dir / 'comparison_report.json'}")
    print(f"Saved: {out_dir / 'comparison_report.md'}")
    if plot_path:
        print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
