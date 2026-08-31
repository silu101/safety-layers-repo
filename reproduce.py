#!/usr/bin/env python3
"""
Unified CLI for this reproduction -- no YAML file needed for a normal run.

This wraps the SAME tested code as the rest of this repo (src/safety_layers_repro/
run_cos_sim.py, run_localization.py, run_harmful_eval.py) -- it does not
reimplement any experiment logic. Every subcommand builds an in-memory
config from your CLI flags (using configs/_empty.yaml as a base, since
every config dataclass field already has a sensible default) and calls the
existing --config-based entry points, so you still get full provenance
(exact resolved config, git commit, timestamp) saved automatically per
run under results/<run_name>/run_metadata.json -- you just never have to
hand-write a YAML file to get it.

If you DO want to save/reuse a specific setup, `--config path/to/some.yaml`
is still available on every subcommand as an optional override base.

Usage examples:
    python reproduce.py existence --model_path meta-llama/Llama-2-7b-chat-hf --r 500
    python reproduce.py localization --model_path google/gemma-2b-it --ranges "6-12,6-13,6-14" --cheng_num 1.1
    python reproduce.py finetune_eval --model_path ./my-finetuned-checkpoint
    python reproduce.py ood ...   # not yet implemented -- see docstring below
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

REPO_ROOT = Path(__file__).resolve().parent
EMPTY_CONFIG = str(REPO_ROOT / "configs" / "_empty.yaml")


def _set_args(**kwargs) -> list[str]:
    """Turns {"model_path": "foo", "r": 500} into ["--set", "model_path=foo",
    "--set", "r=500"], skipping any key whose value is None (so unset CLI
    flags fall through to the dataclass's own default instead of being
    forced to the literal string "None")."""
    out = []
    for k, v in kwargs.items():
        if v is None:
            continue
        out += ["--set", f"{k}={v}"]
    return out


def cmd_existence(args: argparse.Namespace) -> None:
    from safety_layers_repro import run_cos_sim

    config = args.config or EMPTY_CONFIG
    argv = ["--config", config] + _set_args(
        model_path=args.model_path,
        r=args.r,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        normal_path=args.normal_path,
        malicious_path=args.malicious_path,
        run_name=args.run_name,
    )
    run_cos_sim.main(argv)


def _parse_ranges(spec: str) -> list[tuple[int, int]]:
    """"6-12,6-13,6-14" -> [(6,12), (6,13), (6,14)]. Matches this
    project's Python-range convention (end exclusive), same as
    scaling.py's range(begin_num, end_num) -- NOT the paper's own
    inclusive table notation. See docs/KNOWN_DISCREPANCIES.md #13 for why
    this distinction matters."""
    ranges = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        start_s, end_s = chunk.split("-")
        ranges.append((int(start_s), int(end_s)))
    return ranges


def cmd_localization(args: argparse.Namespace) -> None:
    from safety_layers_repro import run_localization

    config = args.config or EMPTY_CONFIG
    ranges = _parse_ranges(args.ranges) if args.ranges else [(args.start_num, args.end_num)]
    cheng_nums = [float(c) for c in args.cheng_num.split(",")] if args.cheng_num else [1.1]

    for start_num, end_num in ranges:
        for cheng_num in cheng_nums:
            print(f"=== localization: range=[{start_num},{end_num}) cheng_num={cheng_num} ===")
            argv = ["--config", config] + _set_args(
                model_path=args.model_path,
                start_num=start_num,
                end_num=end_num,
                cheng_num=cheng_num,
                dtype=args.dtype,
                weight_style=args.weight_style,
                exclude_o_proj=args.exclude_o_proj,
                over_rejection_path=args.over_rejection_path,
                max_prompts=args.max_prompts,
            )
            run_localization.main(argv)


def cmd_finetune_eval(args: argparse.Namespace) -> None:
    from safety_layers_repro import run_harmful_eval

    config = args.config or EMPTY_CONFIG
    argv = ["--config", config] + _set_args(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
        advbench_path=args.advbench_path,
        max_prompts=args.max_prompts,
        use_zou_keyword_classifier=args.use_zou,
        use_harmbench_classifier=args.use_harmbench,
        judge_model=args.judge_model,
        judge_max_samples=args.judge_max_samples,
    )
    run_harmful_eval.main(argv)


def _load_prompt_lines(path: str) -> list[str]:
    with open(path) as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def _write_prompt_lines(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for p in prompts:
            # CSV with a single unquoted column -- matches
            # data/over_rejection.csv / advbench_malicious.csv's own
            # format exactly, so downstream loaders need no changes. This
            # only breaks if a perturbed prompt itself contains a literal
            # newline (paraphrase/templates here don't produce one).
            f.write(p.replace("\n", " ") + "\n")


def cmd_ood(args: argparse.Namespace) -> None:
    """OOD robustness test, per Safe Neuron's own stated recipe (slide 149):
    paraphrase/inject/roleplay the prompt set, then re-run the SAME
    identification (localization) or evaluation (finetune_eval) pipeline
    against the perturbed set. Not a new metric -- reuses existing, tested
    code, just pointed at a perturbed data file instead of the original."""
    import os

    from safety_layers_repro import ood_perturb

    default_source = {
        "localization": "data/over_rejection.csv",
        "finetune_eval": "data/advbench_malicious.csv",
    }[args.target]
    source_path = args.source_path or default_source

    ood_dir = REPO_ROOT / "data" / "ood"
    perturbed_path = ood_dir / f"{args.perturbation}_{Path(source_path).stem}.csv"

    if perturbed_path.exists() and not args.force_regenerate:
        print(f"[ood] Using cached perturbed prompts: {perturbed_path}")
    else:
        prompts = _load_prompt_lines(source_path)
        print(f"[ood] Generating {args.perturbation} variants of {len(prompts)} prompts from {source_path}...")
        client = None
        if args.perturbation == "paraphrase":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise SystemExit("paraphrase needs ANTHROPIC_API_KEY set in the environment.")
            client = anthropic.Anthropic(api_key=api_key)
        perturbed = ood_perturb.perturb(
            args.perturbation, prompts, client=client, model=args.judge_model, max_samples=args.max_prompts,
        )
        _write_prompt_lines(perturbed_path, perturbed)
        print(f"[ood] Wrote {len(perturbed)} perturbed prompts -> {perturbed_path}")

    print(f"[ood] Re-running '{args.target}' against the perturbed set...")
    if args.target == "localization":
        cmd_localization(argparse.Namespace(
            model_path=args.model_path, dtype=args.dtype, config=args.config,
            ranges=args.ranges, start_num=0, end_num=0, cheng_num=args.cheng_num,
            weight_style=args.weight_style, exclude_o_proj=args.exclude_o_proj,
            over_rejection_path=str(perturbed_path), max_prompts=None,
        ))
    else:
        cmd_finetune_eval(argparse.Namespace(
            model_path=args.model_path, dtype=args.dtype, config=args.config,
            tokenizer_path=args.tokenizer_path, advbench_path=str(perturbed_path),
            max_prompts=None, use_zou=args.use_zou, use_harmbench=args.use_harmbench,
            judge_model=args.judge_model, judge_max_samples=args.judge_max_samples,
        ))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--model_path", default=None, help="HF hub id or local checkpoint path.")
        sp.add_argument("--dtype", default=None, choices=["auto", "float16", "bfloat16", "float32"])
        sp.add_argument("--config", default=None, help="Optional YAML base config (overridden by any flags above/below it).")

    p_exist = sub.add_parser("existence", help="Section 3.2-3.3: cosine-similarity existence check.")
    add_common(p_exist)
    p_exist.add_argument("--r", type=int, default=None, help="Draws per pair-type (paper: 500).")
    p_exist.add_argument("--trust_remote_code", action="store_true", default=None)
    p_exist.add_argument("--normal_path", default=None)
    p_exist.add_argument("--malicious_path", default=None)
    p_exist.add_argument("--run_name", default=None)
    p_exist.set_defaults(func=cmd_existence)

    p_loc = sub.add_parser("localization", help="Section 3.4: safety-layer localization sweep.")
    add_common(p_loc)
    p_loc.add_argument("--ranges", default=None, help='e.g. "6-12,6-13,6-14" (comma-separated start-end pairs, end exclusive).')
    p_loc.add_argument("--start_num", type=int, default=0, help="Used only if --ranges is not given.")
    p_loc.add_argument("--end_num", type=int, default=0, help="Used only if --ranges is not given.")
    p_loc.add_argument("--cheng_num", default=None, help='Scaling factor(s), e.g. "1.1" or "1.1,1.15,1.2" to sweep.')
    p_loc.add_argument("--weight_style", default=None, choices=["llama", "phi3"])
    p_loc.add_argument("--exclude_o_proj", action="store_true", default=None,
                        help="Match the actual paper-submission scaling bug -- see docs/KNOWN_DISCREPANCIES.md #17.")
    p_loc.add_argument("--over_rejection_path", default=None)
    p_loc.add_argument("--max_prompts", type=int, default=None)
    p_loc.set_defaults(func=cmd_localization)

    p_ft = sub.add_parser("finetune_eval", help="Section 4: harmful-rate (R_h) / harmful-score (S_h) on AdvBench.")
    add_common(p_ft)
    p_ft.add_argument("--tokenizer_path", default=None, help="Defaults to --model_path if unset.")
    p_ft.add_argument("--advbench_path", default=None)
    p_ft.add_argument("--max_prompts", type=int, default=None)
    p_ft.add_argument("--use_zou", action="store_true", default=None)
    p_ft.add_argument("--use_harmbench", action="store_true", default=None)
    p_ft.add_argument("--judge_model", default=None)
    p_ft.add_argument("--judge_max_samples", type=int, default=None)
    p_ft.set_defaults(func=cmd_finetune_eval)

    p_ood = sub.add_parser(
        "ood",
        help="Out-of-distribution robustness test: paraphrase/inject/roleplay the prompt set, "
             "then re-run localization or finetune_eval against it.",
    )
    add_common(p_ood)
    p_ood.add_argument("--target", required=True, choices=["localization", "finetune_eval"],
                        help="Which existing pipeline to re-run against the perturbed prompts.")
    p_ood.add_argument("--perturbation", required=True, choices=["paraphrase", "inject", "roleplay"])
    p_ood.add_argument("--source_path", default=None,
                        help="Defaults to data/over_rejection.csv (localization) or "
                             "data/advbench_malicious.csv (finetune_eval).")
    p_ood.add_argument("--max_prompts", type=int, default=None, help="Limit how many prompts get perturbed.")
    p_ood.add_argument("--force_regenerate", action="store_true",
                        help="Regenerate perturbed prompts even if a cached file already exists.")
    p_ood.add_argument("--judge_model", default="claude-haiku-4-5-20251001",
                        help="Used both for paraphrase generation and (if --target finetune_eval) S_h judging.")
    # localization-specific pass-through:
    p_ood.add_argument("--ranges", default=None)
    p_ood.add_argument("--cheng_num", default=None)
    p_ood.add_argument("--weight_style", default=None, choices=["llama", "phi3"])
    p_ood.add_argument("--exclude_o_proj", action="store_true", default=None)
    # finetune_eval-specific pass-through:
    p_ood.add_argument("--tokenizer_path", default=None)
    p_ood.add_argument("--use_zou", action="store_true", default=None)
    p_ood.add_argument("--use_harmbench", action="store_true", default=None)
    p_ood.add_argument("--judge_max_samples", type=int, default=None)
    p_ood.set_defaults(func=cmd_ood)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
