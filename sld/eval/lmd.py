import numpy as np
import re
import os
import sys
from functools import partial

from .utils import p, singular, predicate_numeracy, predicate_numeracy_2obj, locations_xywh, predicate_spatial, predicate_attribution, word_to_num_mapping

prompt_prefix = "A realistic photo of a scene"

evaluate_classes = ['backpack', 'book', 'bottle', 'bowl', 'car', 'cat', 'chair', 'cup', 'dog', 'laptop']


def get_prompt_predicates_negation(repeat=10):
    prompt_predicates = []
    
    number = 0
    for object_name in evaluate_classes:
        query_names = (object_name,)

        if prompt_prefix:
            prompt = f"{prompt_prefix} without {p.plural(object_name)}"
        else:
            prompt = f"without {p.plural(object_name)}"
        prompt = prompt.strip()

        prompt_predicate = prompt, partial(predicate_numeracy, query_names, number)
        prompt_predicates += [prompt_predicate] * repeat
    
    return prompt_predicates


def get_prompt_predicates_numeracy(min_num=1, max_num=5, repeat=2):
    prompt_predicates = []

    for number in range(min_num, max_num + 1):
        for object_name in evaluate_classes:
            query_names = (object_name,)

            if prompt_prefix:
                prompt = f"{prompt_prefix} with {p.number_to_words(number) if number < 21 else number} {p.plural(object_name) if number > 1 else object_name}"
            else:
                prompt = f"{p.number_to_words(number) if number < 21 else number} {p.plural(object_name) if number > 1 else object_name}"
            prompt = prompt.strip()

            prompt_predicate = prompt, partial(predicate_numeracy, query_names, number)
            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


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


if __name__ == "__main__":
    # Example usage
    prompts = get_prompt_predicates_negation(repeat=10)
    for prompt, predicate in prompts:
        print(f"Prompt: {prompt}")
        print(f"Predicate: {predicate}")
        print()
