"""
The official refusal_direction pipeline (pipeline.run_pipeline) never
saves a persisted, modified checkpoint -- verified directly by reading
run_pipeline.py: it only saves the extracted direction (direction.pt,
direction_metadata.json) and completion/evaluation JSONs. The
orthogonalization (a permanent weight edit, Table 2's ORTHO method -- see
the paper's Eq. 5) only ever gets applied to the in-memory model object
for their own eval loop, then discarded.

This script applies that same weight edit and actually persists it, so
the resulting checkpoint can be evaluated by handoff_asr_eval/run_asr.py
exactly like a Safety Layers checkpoint -- same HarmBench-classifier ASR
metric, directly comparable across methods.

Must be run from a location where `pipeline` (the refusal_direction
repo's own package) is importable -- either from inside the cloned repo,
or with it on PYTHONPATH.

Usage (after running pipeline.run_pipeline, which produces direction.pt
and direction_metadata.json under pipeline/runs/<model_alias>/):
    cd refusal_direction
    python /path/to/save_orthogonalized_checkpoint.py \\
        --model_path meta-llama/Meta-Llama-3-8B-Instruct \\
        --model_alias meta-llama-3-8b-instruct \\
        --output_dir ./output_model_orthogonalized
"""
import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True, help="Same model_path passed to pipeline.run_pipeline")
    ap.add_argument("--model_alias", required=True,
                     help="The model_alias the pipeline used -- determines pipeline/runs/<model_alias>/ artifact location")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(Path.cwd()))
    try:
        from pipeline.model_utils.model_factory import construct_model_base
    except ImportError:
        raise SystemExit(
            "Couldn't import `pipeline` -- run this script from inside the cloned "
            "refusal_direction repo (or add it to PYTHONPATH)."
        )

    import torch

    artifact_dir = Path("pipeline") / "runs" / args.model_alias
    direction_path = artifact_dir / "direction.pt"
    metadata_path = artifact_dir / "direction_metadata.json"
    if not direction_path.exists():
        raise SystemExit(
            f"{direction_path} not found -- run `python3 -m pipeline.run_pipeline "
            f"--model_path {args.model_path}` first."
        )

    direction = torch.load(direction_path)
    metadata = json.load(open(metadata_path))
    print(f"Loaded direction from layer={metadata['layer']}, pos={metadata['pos']}")

    print(f"Loading model: {args.model_path}")
    model_base = construct_model_base(args.model_path)

    print("Applying orthogonalization (permanent weight edit)...")
    # Verified directly against pipeline/model_utils/llama3_model.py:
    # _get_orthogonalization_mod_fn(direction) returns
    # functools.partial(orthogonalize_llama3_weights, direction=direction),
    # which mutates model.model.embed_tokens / each layer's o_proj and
    # mlp.down_proj weights IN PLACE when called with the model. Other
    # model families (gemma_model.py, qwen_model.py, etc.) follow the same
    # pattern -- construct_model_base() already picked the right one for
    # you based on model_path.
    orthogonalization_fn = model_base._get_orthogonalization_mod_fn(direction)
    orthogonalization_fn(model=model_base.model)

    print(f"Saving orthogonalized checkpoint -> {args.output_dir}")
    model_base.model.save_pretrained(args.output_dir)
    model_base.tokenizer.save_pretrained(args.output_dir)
    print("Done. Now run handoff_asr_eval/run_asr.py against this checkpoint.")


if __name__ == "__main__":
    main()
