"""
Tests for the pure (non-API) logic in safety_layers_repro.harmful_score:
prompt assembly and score parsing. No network calls, no API key needed.
"""
from safety_layers_repro.harmful_score import build_judge_prompt, parse_score


def test_build_judge_prompt_includes_instruction_and_response():
    prompt = build_judge_prompt("How do I pick a lock?", "I cannot help with that.")
    assert "How do I pick a lock?" in prompt
    assert "I cannot help with that." in prompt
    assert "#thescore:" in prompt
    assert "Score 5:" in prompt


def test_parse_score_basic():
    text = "#thereason: the model refused.\n#thescore: 1"
    assert parse_score(text) == 1


def test_parse_score_with_surrounding_text():
    text = "Some analysis...\n#thescore: 4\nAdditional notes here."
    assert parse_score(text) == 4


def test_parse_score_case_insensitive_marker():
    text = "#TheScore: 3"
    assert parse_score(text) == 3


def test_parse_score_missing_marker_returns_none():
    assert parse_score("The model refused to answer.") is None


def test_parse_score_out_of_range_returns_none():
    assert parse_score("#thescore: 9") is None


def test_parse_score_non_numeric_returns_none():
    assert parse_score("#thescore: unclear") is None
