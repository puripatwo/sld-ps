import argparse
import configparser

import os
import json
import copy
import shutil
import random
import numpy as np
from PIL import Image

import torch
import diffusers

import models
from models import sam
from models.detector import OWLVITV2Detector
from utils import parse, utils

from llm.llm_parser_template import spot_object_template
from llm.llm_controller_template import spot_difference_template
from llm.llm_chat import get_key_objects, get_updated_layout


def spot_objects(prompt, data, config):
    if data.get("llm_parsed_prompt") is None:
        questions = f"User Prompt: {prompt}\nReasoning:\n"
        message = spot_object_template + questions
        results = get_key_objects(message, config)
        return results[0]
    else:
        return data["llm_parsed_prompt"]


def spot_differences(prompt, det_results, data, config, mode="self_correction"):
    if data.get("llm_layout_suggestions") is None:
        questions = (f"User Prompt: {prompt}\nCurrent Objects: {det_results}\nReasoning:\n")
        message = spot_difference_template + questions
        llm_suggestions = get_updated_layout(message, config)
        return llm_suggestions[0]
    else:
        return data["llm_layout_suggestions"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLD")
    parser.add_argument("--json-file", type=str, default="data/data.json", help="Path to data.json")
    parser.add_argument("--input-dir", type=str, default="data/input_dir", help="Path to the input directory")
    parser.add_argument("--output-dir", type=str, default="data/output_dir", help="Path to the output directory")
    parser.add_argument("--mode", type=str, default="self_correction", help="Mode of the demo", choices=["self_correction", "image_editing"])
    parser.add_argument("--config", type=str, default="sld_config.ini", help="Path to the config file")
    args = parser.parse_args()

    # Load the json file
    with open(args.json_file) as f:
        data = json.load(f)
    save_dir = args.output_dir
    parse.img_dir = os.path.join(save_dir, "tmp_imgs") # parse.img_dir = data/output_dir/tmp_imgs
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(parse.img_dir, exist_ok=True)

    # Load the config file
    config = configparser.ConfigParser()
    config.read(args.config)

    # Load the models
    models.sd_key = "gligen/diffusers-generation-text-box"
    models.sd_version = "sdv1.4"
    diffusion_scheduler = None
    models.model_dict = models.load_sd(
        key=models.sd_key,
        use_fp16=False,
        load_inverse_scheduler=True,
        use_dpm_multistep_scheduler=False,
        scheduler_cls=diffusers.schedulers.__dict__[diffusion_scheduler] if diffusion_scheduler is not None else False,
    )
    # sam_model_dict = sam.load_sam()
    # models.model_dict.update(sam_model_dict)
    # det = OWLVITV2Detector()

    for idx in range(len(data)):
        # Reset random seeds
        default_seed = int(config.get("SLD", "default_seed"))
        torch.manual_seed(default_seed)
        np.random.seed(default_seed)
        random.seed(default_seed)

        # Load the image and prompt
        input_fname = data[idx]["input_fname"]
        fname = os.path.join(args.input_dir, f"{input_fname}.png") # fname = data/input_dir/{input_fname}.png
        prompt = data[idx]["prompt"]

        # Create an output directory for the current image
        dirname = os.path.join(save_dir, data[idx]["output_dir"]) # dirname = data/output_dir/{output_dir}
        os.makedirs(dirname, exist_ok=True)
        output_fname = os.path.join(dirname, f"initial_image.png") # output_fname = data/output_dir/{output_dir}/initial_image.png
        shutil.copy(fname, output_fname)

        print("-" * 5 + f" [Self-Correcting {fname}] " + "-" * 5)
        print(f"Target Textual Prompt: {prompt}")

        # Step 1: Spot Objects with LLM
        print("-" * 5 + f" Parsing Prompts " + "-" * 5)
        llm_parsed_prompt = spot_objects(prompt, data[idx], config)
        entry = {"instructions": prompt, "output": [fname],
                 "generator": data[idx]["generator"],
                 "objects": llm_parsed_prompt["objects"], 
                 "bg_prompt": llm_parsed_prompt["bg_prompt"],
                 "neg_prompt": llm_parsed_prompt["neg_prompt"]
                }
        print(f"* Objects: {entry['objects']}")
        print(f"* Background: {entry['bg_prompt']}")
        print(f"* Negation: {entry['neg_prompt']}")

        # # Step 2: Run open vocabulary detector
        # print("-" * 5 + f" Running Detector " + "-" * 5)
        # default_attr_threshold = float(config.get("SLD", "attr_detection_threshold")) 
        # default_prim_threshold = float(config.get("SLD", "prim_detection_threshold"))
        # default_nms_threshold = float(config.get("SLD", "nms_threshold"))

        # attr_threshold = float(config.get(entry["generator"], "attr_detection_threshold", fallback=default_attr_threshold))
        # prim_threshold = float(config.get(entry["generator"], "prim_detection_threshold", fallback=default_prim_threshold))
        # nms_threshold = float(config.get(entry["generator"], "nms_threshold", fallback=default_nms_threshold))

        # det_results = det.run(prompt, entry["objects"], entry["output"][-1],
        #                       attr_detection_threshold=attr_threshold, 
        #                       prim_detection_threshold=prim_threshold, 
        #                       nms_threshold=nms_threshold)
        
        # # Step 3: Spot difference between detected results and initial prompts
        # print("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)
        # llm_suggestions = spot_differences(prompt, det_results, data[idx], config, mode=args.mode)
        # entry["det_results"] = copy.deepcopy(det_results)
        # entry["llm_suggestions"] = llm_suggestions
        # print(f"* Detection Restuls: {det_results}")
        # print(f"* LLM Suggestions: {llm_suggestions}")

        # # Step 4: Check which objects to preserve, delete, add, reposition, or modify
        # print("-" * 5 + f" Editing Operations " + "-" * 5)
        # (
        #     preserve_objs,
        #     deletion_objs,
        #     addition_objs,
        #     repositioning_objs,
        #     attr_modification_objs,
        # ) = det.parse_list(det_results, llm_suggestions)
        # print(f"* Preservation: {preserve_objs}")
        # print(f"* Addition: {addition_objs}")
        # print(f"* Deletion: {deletion_objs}")
        # print(f"* Repositioning: {repositioning_objs}")
        # print(f"* Attribute Modification: {attr_modification_objs}")

        # # Step 5: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
        