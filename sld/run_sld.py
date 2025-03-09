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
from models.sam import run_sam, run_sam_postprocess
from models.pipelines import get_all_latents
from models.detector import OWLVITV2Detector

from utils import parse, utils, resize_image

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


# Operation #2: Deletion (Preprocessing region mask for removal)
def get_remove_region(entry, remove_objects, move_objects, preserve_objs, models, config):
    """Generate a region mask for removal given bounding box info."""

    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape

    # if no remove objects, set zero to the whole mask
    if (len(remove_objects) + len(move_objects)) == 0:
        remove_region = np.zeros((W // 8, H // 8), dtype=np.int64)
        return remove_region
    
    # Otherwise, run the SAM segmentation to locate target regions
    remove_items = remove_objects + [x[0] for x in move_objects]
    remove_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in remove_items:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        remove_mask = remove_mask | masks

    # Preserve the regions that should not be removed
    preserve_mask = np.zeros((H, W, 3), dtype=bool)
    for obj in preserve_objs:
        masks = run_sam(bbox=obj[1], image_source=image_source, models=models)
        preserve_mask = preserve_mask | masks
    # Process the SAM mask by averaging, thresholding, and dilating.
    preserve_region = run_sam_postprocess(preserve_mask, H, W, config)
    remove_region = run_sam_postprocess(remove_mask, H, W, config)
    remove_region = np.logical_and(remove_region, np.logical_not(preserve_region))
    return remove_region


# Operation #3: Repositioning (Preprocessing latent)
def get_repos_info(entry, move_objects, models, config):
    """
    Updates a list of objects to be moved / reshaped, including resizing images and generating masks.
    * Important: Perform image reshaping at the image-level rather than the latent-level.
    * Warning: For simplicity, the object is not positioned to the center of the new region...
    """

    # if no remove objects, set zero to the whole mask
    if not move_objects:
        return move_objects
    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    new_move_objects = []
    for item in move_objects:
        new_img, obj = resize_image(image_source, item[0][1], item[1][1])
        masks = run_sam(obj, new_img, models)
        old_object_region = run_sam_postprocess(masks, H, W, config).astype(np.bool_)
        all_latents, _ = get_all_latents(new_img, models, inv_seed)
        new_move_objects.append(
            [item[0][0], obj, item[1][1], old_object_region, all_latents]
        )

    return new_move_objects


# Operation #4: Attribute Modification (Preprocessing latent)
def get_attrmod_latent(entry, change_attr_objects, models, config):
    if len(change_attr_objects) == 0:
        return []
    
    from diffusers import StableDiffusionDiffEditPipeline
    from diffusers import DDIMScheduler, DDIMInverseScheduler

    img = Image.open(entry["output"][-1])
    image_source = np.array(img)
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))

    # Initialize the Stable Diffusion pipeline
    pipe = StableDiffusionDiffEditPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
    pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    new_change_objects = []
    for obj in change_attr_objects:
        # Run diffedit
        old_object_region = run_sam_postprocess(run_sam(obj[1], image_source, models), H, W, config)
        old_object_region = old_object_region.astype(np.bool_)[np.newaxis, ...]

        new_object = obj[0].split(" #")[0]
        base_object = new_object.split(" ")[-1]
        mask_prompt = f"a {base_object}"
        new_prompt = f"a {new_object}"

        image_latents = pipe.invert(
            image=img,
            prompt=mask_prompt,
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
        ).latents
        image = pipe(
            prompt=new_prompt,
            mask_image=old_object_region,
            image_latents=image_latents,
            guidance_scale=float(config.get("SLD", "diffedit_guidance_scale")),
            inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            negative_prompt="",
        ).images[0]

        all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
        new_change_objects.append(
            [
                old_object_region[0],
                all_latents,
            ]
        )
    return new_change_objects


def correction(entry, add_objects, move_objects, remove_region, change_attr_objects, models, config):
    spec = {
        "add_objects": add_objects,
        "move_objects": move_objects,
        "prompt": entry["instructions"],
        "remove_region": remove_region,
        "change_objects": change_attr_objects,
        "all_objects": entry["llm_suggestion"],
        "bg_prompt": entry["bg_prompt"],
        "extra_neg_prompt": entry["neg_prompt"],
    }
    image_source = np.array(Image.open(entry["output"][-1]))

    # Run the correction pipeline
    all_latents, _ = get_all_latents(image_source, models, int(config.get("SLD", "inv_seed")))
    ret_dict = image_generator.run(
        spec,
        fg_seed_start=int(config.get("SLD", "fg_seed")),
        bg_seed=int(config.get("SLD", "bg_seed")),
        bg_all_latents=all_latents,
        frozen_step_ratio=float(config.get("SLD", "frozen_step_ratio")),
    )

    return ret_dict


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
        scheduler_cls=diffusers.schedulers.__dict__[diffusion_scheduler] if diffusion_scheduler is not None else None,
    )
    sam_model_dict = sam.load_sam()
    models.model_dict.update(sam_model_dict)
    det = OWLVITV2Detector()

    from models import image_generator

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

        # Step 2: Run open vocabulary detector
        print("-" * 5 + f" Running Detector " + "-" * 5)
        default_attr_threshold = float(config.get("SLD", "attr_detection_threshold")) 
        default_prim_threshold = float(config.get("SLD", "prim_detection_threshold"))
        default_nms_threshold = float(config.get("SLD", "nms_threshold"))

        attr_threshold = float(config.get(entry["generator"], "attr_detection_threshold", fallback=default_attr_threshold))
        prim_threshold = float(config.get(entry["generator"], "prim_detection_threshold", fallback=default_prim_threshold))
        nms_threshold = float(config.get(entry["generator"], "nms_threshold", fallback=default_nms_threshold))

        det_results = det.run(prompt, entry["objects"], entry["output"][-1],
                              attr_detection_threshold=attr_threshold, 
                              prim_detection_threshold=prim_threshold, 
                              nms_threshold=nms_threshold)
        
        # Step 3: Spot difference between detected results and initial prompts
        print("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)
        llm_suggestions = spot_differences(prompt, det_results, data[idx], config, mode=args.mode)
        entry["det_results"] = copy.deepcopy(det_results)
        entry["llm_suggestions"] = copy.deepcopy(llm_suggestions)
        print(f"* Detection Results: {det_results}")
        print(f"* LLM Suggestions: {llm_suggestions}")

        # Step 4: Check which objects to preserve, delete, add, reposition, or modify
        print("-" * 5 + f" Editing Operations " + "-" * 5)
        (
            preserve_objs,
            deletion_objs,
            addition_objs,
            repositioning_objs,
            attr_modification_objs,
        ) = det.parse_list(det_results, llm_suggestions)
        print(f"* Preservation: {preserve_objs}")
        print(f"* Addition: {addition_objs}")
        print(f"* Deletion: {deletion_objs}")
        print(f"* Repositioning: {repositioning_objs}")
        print(f"* Attribute Modification: {attr_modification_objs}")

        # Visualize the detection results
        parse.show_boxes(
            gen_boxes=entry["det_results"],
            additional_boxes=entry["llm_suggestions"],
            img=np.array(Image.open(entry["output"][-1])).astype(np.uint8),
            fname=os.path.join(dirname, "det_result_obj.png"),
        )

        # Check if there are any operations to perform
        total_ops = len(deletion_objs) + len(addition_objs) + len(repositioning_objs) + len(attr_modification_objs)
        if total_ops == 0:
            print("-" * 5 + f" Results " + "-" * 5)
            output_fname = os.path.join(dirname, f"final_image.png") # output_fname = data/output_dir/{output_dir}/final_image.png
            shutil.copy(entry["output"][-1], output_fname)
            print(f"No operations needed. The final image is saved at {output_fname}")
            continue

        # Step 5: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
        print("-" * 5 + f" Image Manipulation " + "-" * 5)

        deletion_region = get_remove_region(
            entry, deletion_objs, repositioning_objs, [], models, config
        )
        print(f"* Deletion: {deletion_region}")
        repositioning_objs = get_repos_info(
            entry, repositioning_objs, models, config
        )
        print(f"* Repositioning: {repositioning_objs}")
        attr_modification_objs = get_attrmod_latent(
            entry, attr_modification_objs, models, config
        )
        print(f"* Attribute Modification: {attr_modification_objs}")
        ret_dict = correction(
            entry, addition_objs, repositioning_objs, deletion_region, attr_modification_objs, models, config
        )

        print()
        