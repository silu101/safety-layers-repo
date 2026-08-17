"""
Prompt template handling.

Adapted from the original repo's Code/utils/prompter.py. Behavior is kept
identical (same template lookup semantics, same generate_prompt/get_response
logic) so that prompts produced here are byte-identical to the original
pipeline given the same template file.
"""
from __future__ import annotations

import json
import os.path as osp
from typing import Optional, Union

_TEMPLATE_DIR = osp.join(osp.dirname(__file__))


class Prompter:
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "alpaca", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "alpaca"
        file_name = osp.join(_TEMPLATE_DIR, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(f"Using prompt template {template_name}: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Optional[str] = None,
        label: Optional[str] = None,
    ) -> str:
        if input:
            res = self.template["prompt_input"].format(instruction=instruction, input=input)
        else:
            res = self.template["prompt_no_input"].format(instruction=instruction)
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
