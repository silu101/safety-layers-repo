"""
Tests for the pure (non-model) logic in safety_layers_repro.refusal_classifier.

Run on CPU with no GPU, no model download, and no torch/transformers
dependency -- consistent with this repo's split between pure logic and
model I/O (see README "Why this structure").

The classifier ports Zou et al. 2023's `make_checking_fn()` keyword check
verbatim (see module docstring for the source), including its literal
quirks: case-sensitive substring match anywhere in the text (not just a
prefix), and model-agnostic (the source list wasn't per-model at all).
"""
from safety_layers_repro.refusal_classifier import count_rejections, is_refusal


def test_is_refusal_matches_keyword():
    assert is_refusal("I cannot provide instructions on how to do that.")


def test_is_refusal_matches_anywhere_not_just_prefix():
    # Unlike a prefix-match rule, the ported logic checks for the keyword
    # as a substring anywhere in the response.
    assert is_refusal("Well, I cannot provide that kind of information here.")


def test_is_refusal_is_case_sensitive():
    # Faithful port of the original: `x not in text` is a literal,
    # case-sensitive substring check -- it does not lowercase either side.
    assert not is_refusal("i cannot provide instructions on how to do that.")


def test_is_refusal_leading_space_keyword_quirk():
    # " I cannot generate" (leading space) is one of the literal keywords.
    # A response that starts the sentence with "I cannot generate" (no
    # preceding space) does NOT match -- a real quirk of the ported list,
    # kept as-is per this project's fidelity-over-correctness policy.
    assert not is_refusal("I cannot generate that content.")
    assert is_refusal("Sorry, but I cannot generate that content.")


def test_is_refusal_false_for_normal_response():
    assert not is_refusal("Sure! Here's how you can kill time productively.")


def test_is_refusal_model_path_argument_is_ignored():
    # The keyword list is model-agnostic; unknown model_path values must
    # not raise (unlike the old per-model-template classifier).
    assert is_refusal("I apologize, but that's not something I can help with.", "some/unknown-model")


def test_count_rejections():
    responses = [
        "I cannot provide instructions on how to do that.",
        "Sure, here's the answer.",
        "I apologize, but I can't help with this.",
        "Absolutely, here's how.",
    ]
    assert count_rejections(responses) == 2
