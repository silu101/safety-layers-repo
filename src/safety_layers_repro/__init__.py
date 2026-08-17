"""
safety_layers_repro
====================

Modular reproduction of the "safety layers" cosine-similarity existence
analysis (Section 3.2-3.3) from:

    Li, Yao, Zhang, Li. "Safety Layers in Aligned Large Language Models:
    The Key to LLM Security." ICLR 2025. https://arxiv.org/abs/2408.17003

Original authors' code: https://github.com/listen0425/Safety-Layers

This package refactors `Code/Cos_sim_analysis/save_all_pairs_cos_sim.py`
from the original repo into a config-driven, testable pipeline, and adds
a numeric comparison tool (`compare.py`) so reproduction can be checked
against a reference result quantitatively, not just by eyeballing plots.
"""

__version__ = "0.1.0"
