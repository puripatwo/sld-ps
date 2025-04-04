import numpy as np
import re
from functools import partial
import os
import sys

from .utils import p, singular, predicate_numeracy, predicate_numeracy_2obj, locations_xywh, predicate_spatial, predicate_attribution, word_to_num_mapping


def get_lmd_prompts():
    prompt_predicates_negation = get_prompt_predicates_negation(repeat=10)
    prompt_predicates_numeracy = get_prompt_predicates_numeracy(max_num=5, repeat=2)
    prompt_predicates_attribution = get_prompt_predicates_attribution(num_prompts=100)
    prompt_predicates_spatial = get_prompt_predicates_spatial(num_prompts=25)

    prompts_negation = [prompt for prompt, _ in prompt_predicates_negation]
    prompts_numeracy = [prompt for prompt, _ in prompt_predicates_numeracy]
    prompts_attribution = [prompt for prompt, _ in prompt_predicates_attribution]
    prompts_spatial = [prompt for prompt, _ in prompt_predicates_spatial]

    prompts_all = prompts_negation + prompts_numeracy + prompts_attribution + prompts_spatial

    prompts = {
        'lmd': prompts_all,
        'lmd_negation': prompts_negation,
        'lmd_numeracy': prompts_numeracy,
        'lmd_attribution': prompts_attribution,
        'lmd_spatial': prompts_spatial,
    }

    return prompts
