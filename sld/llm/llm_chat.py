# Apply self-correcting LLM-controlled diffusion (SLD)
# Reference: Wu et al., "Self-Correcting LLM-Controlled Diffusion Models"
# https://arxiv.org/abs/2309.16668

import ast
import time
from openai import OpenAI

def get_key_objects(message, config):
    # Reading configuration from file
    organization = config.get("openai", "organization")
    api_key = config.get("openai", "api_key")
    model = config.get("openai", "model")

    messages = [{"role": "user", "content": message}]
    while True:
        try:
            client = OpenAI(organization=organization, api_key=api_key)
            response = client.chat.completions.create(model=model, messages=messages)
            raw_response = response.choices[0].message.content
            print(f"ChatGPT: {raw_response}")

            # Extracting key objects
            key_objects_part = raw_response.split("Objects:")[1]
            start_index = key_objects_part.index("[")
            end_index = key_objects_part.rindex("]") + 1
            objects_str = key_objects_part[start_index:end_index]

            # Converting string to list
            parsed_objects = ast.literal_eval(objects_str)

            # Extracting additional negative prompt
            bg_prompt = raw_response.split("Background:")[1].split("\n")[0].strip()
            negative_prompt = raw_response.split("Negation:")[1].strip()
            break
        except Exception as e:
            print(f"Error occured when calling LLM API: {e}")
            time.sleep(5)

    parsed_result = {
        "objects": parsed_objects,
        "bg_prompt": bg_prompt,
        "neg_prompt": negative_prompt,
    }
    return parsed_result, raw_response

def get_updated_layout(message, config):
    # Reading configuration from file
    organization = config.get("openai", "organization")
    api_key = config.get("openai", "api_key")
    model = config.get("openai", "model")

    messages = [{"role": "user", "content": message}]
    while True:
        try:
            client = OpenAI(organization=organization, api_key=api_key)
            response = client.chat.completions.create(model=model, messages=messages)
            raw_response = response.choices[0].message.content
            print(f"ChatGPT: {raw_response}")

            # Extracting bounding box data
            bbox_data = raw_response.split("Updated Objects:")[1]
            start_index = bbox_data.index("[")
            end_index = bbox_data.rindex("]") + 1
            bbox_str = bbox_data[start_index:end_index]

            # Converting string to list
            updated_bboxes = ast.literal_eval(bbox_str)
            break
        except Exception as e:
            print(f"Error occured when calling LLM API: {e}")
            time.sleep(5)

    return updated_bboxes, raw_response
