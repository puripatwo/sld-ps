import numpy as np
import re
import os
import sys
from functools import partial

from .utils import p, predicate_numeracy, predicate_numeracy_2obj,predicate_spatial, predicate_attribution, singular, locations_xywh, word_to_num_mapping

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


def process_object_name(object_name):
    if isinstance(object_name, tuple):
        query_names = object_name
        object_name = object_name[0]
    else:
        query_names = (object_name,)
    
    return object_name, query_names


def get_prompt_predicates_attribution(num_prompts=100, repeat=1):
    prompt_predicates = []

    modifiers = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink', 'brown', 'black', 'white', 'gray']
    evaluate_classes_np = np.array(evaluate_classes, dtype=object)
    intended_count1, intended_count2 = 1, 1

    for i in range(num_prompts):
        np.random.seed(i)
        modifier1, modifier2 = np.random.choice(modifiers, 2, replace=False)
        object_name1, object_name2 = np.random.choice(evaluate_classes_np, 2, replace=False)

        object_name1, query_names1 = process_object_name(object_name1)
        object_name2, query_names2 = process_object_name(object_name2)

        if prompt_prefix:
            prompt = f"{prompt_prefix} with {p.a(modifier1)} {object_name1} and {p.a(modifier2)} {object_name2}"
        else:
            prompt = f"{p.a(modifier1)} {object_name1} and {p.a(modifier2)} {object_name2}"
        prompt = prompt.strip()

        prompt_predicate = prompt, partial(predicate_attribution, query_names1, query_names2, modifier1, modifier2, intended_count1, intended_count2)
        prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_prompt_predicates_spatial(num_prompts=25, left_right_only=False):
    prompt_predicates = []

    evaluate_classes_np = np.array(evaluate_classes, dtype=object)
    repeat = 1

    # The boxes are in (x, y, w, h) format
    locations = [
        ('left', 'right', lambda box1, box2: box1[0] + box1[2]/2 < box2[0] + box2[2]/2),
        ('right', 'left', lambda box1, box2: box1[0] + box1[2]/2 > box2[0] + box2[2]/2),
    ]
    if not left_right_only:
        locations += [
            ('top', 'bottom', lambda box1, box2: box1[1] + box1[3]/2 < box2[1] + box2[3]/2),
            ('bottom', 'top', lambda box1, box2: box1[1] + box1[3]/2 > box2[1] + box2[3]/2)
        ]

    for i in range(num_prompts):
        np.random.seed(i)
        for location1, location2, verify_fn in locations:
            object_name1, object_name2 = np.random.choice(evaluate_classes_np, 2, replace=False)

            object_name1, query_names1 = process_object_name(object_name1)
            object_name2, query_names2 = process_object_name(object_name2)

            if prompt_prefix:
                prompt = f"{prompt_prefix} with {p.a(object_name1)} on the {location1} and {p.a(object_name2)} on the {location2}"
            else:
                prompt = f"{p.a(object_name1)} on the {location1} and {p.a(object_name2)} on the {location2}"
            prompt = prompt.strip()

            prompt_predicate = prompt, partial(predicate_spatial, query_names1, query_names2, verify_fn)
            prompt_predicates += [prompt_predicate] * repeat

    return prompt_predicates


def get_eval_info_from_prompt_lmd(prompt):
    if 'without' in prompt:
        # 1. Negation
        pattern = f"without (.+)"
        match = re.search(pattern, prompt)
        object_name = match.group(1)
        object_name = singular(object_name)
        texts = [[f"image of {p.a(object_name)}"]]
        query_names = (object_name,)
        number = 0
        predicate = partial(predicate_numeracy, query_names, number)
        eval_info = {"type": "negation", "predicate": predicate}
    elif 'on the left' in prompt or 'on the right' in prompt or 'on the top' in prompt or 'on the bottom' in prompt:
        # 2. Spatial
        pattern = f"with (.+) on the (.+) and (.+) on the (.+)"
        match = re.search(pattern, prompt)
        print("prompt:", prompt)
        object_name1, location1 = match.group(1), match.group(2)
        object_name2, location2 = match.group(3), match.group(4)
        texts = [[f"image of {object_name1}", f"image of {object_name2}"]]
        query_names1, query_names2 = (object_name1, ), (object_name2, )
        verify_fn = locations_xywh[(location1, location2)]
        predicate = partial(predicate_spatial, query_names1, query_names2, verify_fn)
        eval_info = {"type": "spatial", "location1": location1, "location2": location2, "predicate": predicate}
    elif 'and' in prompt:
        if 'one' in prompt or 'two' in prompt or 'three' in prompt or 'four' in prompt or 'five' in prompt:
            # 3. Numeracy (2 objects)
            pattern = f"with (.+) (.+) and (.+) (.+)"
            match = re.search(pattern, prompt)
            number1, object_name1 = match.group(1), match.group(2)
            number2, object_name2 = match.group(1), match.group(2)
            
            number1 = word_to_num_mapping[number1] if number1 in word_to_num_mapping else int(number1)
            number2 = word_to_num_mapping[number2] if number2 in word_to_num_mapping else int(number2)
            
            object_name1, object_name2 = singular(object_name1), singular(object_name2)
            texts = [[f"image of {p.a(object_name1)}", f"image of {p.a(object_name2)}"]]
            query_names1, query_names2 = (object_name1,), (object_name2,)
            predicate = partial(predicate_numeracy_2obj, query_names1, number1, query_names2, number2)
            eval_info = {"type": "numeracy_2obj", "object_name1": object_name1, "number1": number1, "object_name2": object_name2, "number2": number2, "predicate": predicate}
        else:
            # 4. Attribution
            assert 'on the' not in prompt, prompt
            pattern = f"with (.+) and (.+)"
            match = re.search(pattern, prompt)
            object_name1 = match.group(1)
            object_name2 = match.group(2)
            texts = [[f"image of {object_name1}", f"image of {object_name2}"]]
            query_names1, query_names2 = (object_name1, ), (object_name2, )
            modifier1, modifier2, intended_count1, intended_count2 = None, None, 1, 1
            predicate = partial(predicate_attribution, query_names1, query_names2, modifier1, modifier2, intended_count1, intended_count2)
            eval_info = {"type": "attribution", "object_name1": object_name1, "object_name2": object_name2, "predicate": predicate}
    elif 'with' in prompt: # with number words
        # 5. Numeracy (1 object)
        pattern = f"with (.+) (.+)"
        match = re.search(pattern, prompt)
        number, object_name = match.group(1), match.group(2)
        
        if number not in word_to_num_mapping:
            number = int(number)
        else:
            number = word_to_num_mapping[number]

        object_name = singular(object_name)
        texts = [[f"image of {p.a(object_name)}"]]
        query_names = (object_name,)
        predicate = partial(predicate_numeracy, query_names, number)
        eval_info = {"type": "numeracy", "object_name": object_name, "number": number, "predicate": predicate}
    else:
        raise ValueError(f"Unknown LMD prompt type: {prompt}")


def get_lmd_prompts():
    prompt_predicates_negation = get_prompt_predicates_negation(repeat=10)
    prompt_predicates_numeracy = get_prompt_predicates_numeracy(max_num=5, repeat=2)
    prompt_predicates_attribution = get_prompt_predicates_attribution(num_prompts=10)
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
    prompts = get_prompt_predicates_negation(repeat=10)
    for prompt, predicate in prompts:
        print(f"Prompt: {prompt}")
        print(f"Predicate: {predicate}")
        print()
