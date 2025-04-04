import numpy as np
import inflect
import re

p = inflect.engine()


def count(gen_boxes, name_include):
    return sum([
        any([name_include_item in box['name'] for name_include_item in name_include]) 
        for box in gen_boxes
    ])


def predicate_numeracy(query_names, intended_count, gen_boxes, verbose=False):
    object_count = count(gen_boxes, name_include=query_names)

    if verbose:
        print(
            f"object_count: {object_count}, intended_count: {intended_count} (gen_boxes: {gen_boxes}, query_names: {query_names})")
        
    return object_count == intended_count
