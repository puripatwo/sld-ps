import argparse
import configparser

import os
import json
import copy
import shutil
import random
import logging
import numpy as np
from PIL import Image
import torch
import diffusers

import models
from models import sam
from models.sam import run_sam, run_sam_postprocess
from models.pipelines import get_all_latents
from models.detector import OWLVITV2Detector
from models.sdxl_refine import sdxl_refine

from utils import parse, utils, resize_image

from llm.llm_parser_template import spot_object_template
from llm.llm_controller_template import spot_difference_template
from llm.llm_chat import get_key_objects, get_updated_layout

from eval.eval import Evaluator, eval_prompt
from eval.lmd import get_lmd_prompts

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)


def generate_image(model_key, prompt, seed, target_size=(512, 512)):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(prompt, generator=generator).images[0]

    # Resize image to 512x512 if needed
    if image.size != target_size:
        image = image.resize(target_size, Image.LANCZOS)

    return image


def run_llm_parser(prompt, config):
    questions = f"User Prompt: {prompt}\nReasoning:\n"
    message = spot_object_template + questions
    results = get_key_objects(message, config)
    return results


def run_llm_controller(prompt, det_results, config, mode="self_correction"):
    questions = (f"User Prompt: {prompt}\nCurrent Objects: {det_results}\nReasoning:\n")
    message = spot_difference_template + questions
    llm_suggestions = get_updated_layout(message, config)
    return llm_suggestions


def set_file_handler(log_file_name):
    # Get the root logger
    logger = logging.getLogger()

    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logger.addHandler(console_handler)

    # Create a file handler
    file_handler = logging.FileHandler(log_file_name, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# Operation #2: Deletion (Preprocessing region mask for removal)
def get_remove_region(entry, remove_objects, move_objects, preserve_objs, models, config):
    """Generate a region mask for removal given bounding box info."""

    image_source = np.array(Image.open(entry["output"][-1]))
    H, W, _ = image_source.shape

    # If there are no objects to be moved, set zero to the whole mask
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

    # If there are no objects to be moved, set zero to the whole mask
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


# # Operation #4: Attribute Modification (Preprocessing latent)
# def get_attrmod_latent(entry, change_attr_objects, models, config):
#     if len(change_attr_objects) == 0:
#         return []
    
#     from diffusers import StableDiffusionDiffEditPipeline
#     from diffusers import DDIMScheduler, DDIMInverseScheduler

#     img = Image.open(entry["output"][-1])
#     image_source = np.array(img)
#     H, W, _ = image_source.shape
#     inv_seed = int(config.get("SLD", "inv_seed"))

#     # Initialize the Stable Diffusion pipeline
#     pipe = StableDiffusionDiffEditPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
#     pipe.inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
#     pipe.enable_model_cpu_offload()

#     new_change_objects = []
#     for obj in change_attr_objects:
#         # Run diffedit
#         old_object_region = run_sam_postprocess(run_sam(obj[1], image_source, models), H, W, config)
#         old_object_region = old_object_region.astype(np.bool_)[np.newaxis, ...]

#         new_object = obj[0].split(" #")[0]
#         base_object = new_object.split(" ")[-1]
#         mask_prompt = f"a {base_object}"
#         new_prompt = f"a {new_object}"

#         image_latents = pipe.invert(
#             image=img,
#             prompt=mask_prompt,
#             inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
#             generator=torch.Generator(device="cuda").manual_seed(inv_seed),
#         ).latents
#         image = pipe(
#             prompt=new_prompt,
#             mask_image=old_object_region,
#             image_latents=image_latents,
#             guidance_scale=float(config.get("SLD", "diffedit_guidance_scale")),
#             inpaint_strength=float(config.get("SLD", "diffedit_inpaint_strength")),
#             generator=torch.Generator(device="cuda").manual_seed(inv_seed),
#             negative_prompt="",
#         ).images[0]

#         all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
#         new_change_objects.append(
#             [
#                 old_object_region[0],
#                 all_latents,
#             ]
#         )
#     return new_change_objects


# Operation #4: Attribute Modification (Preprocessing latent)
def get_attrmod_latent(entry, change_attr_objects, models, config):
    if len(change_attr_objects) == 0:
        return []

    from diffusers import StableDiffusionDiffEditPipeline
    from diffusers import DDIMScheduler, DDIMInverseScheduler

    image_path = entry["output"][-1]
    img = Image.open(image_path).convert("RGB")
    image_source = np.array(img)
    H, W, _ = image_source.shape
    inv_seed = int(config.get("SLD", "inv_seed"))
    inpaint_strength = float(config.get("SLD", "diffedit_inpaint_strength"))
    guidance_scale = float(config.get("SLD", "diffedit_guidance_scale"))
    num_inference_steps = int(config.get("SLD", "num_inference_steps", fallback="41"))

    # Initialize the pipeline and replace both schedulers
    pipe = StableDiffusionDiffEditPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        torch_dtype=torch.float16
    ).to("cuda")

    # Use custom schedulers and explicitly set timesteps
    scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inverse_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)

    scheduler.set_timesteps(num_inference_steps)
    inverse_scheduler.set_timesteps(num_inference_steps)

    pipe.scheduler = scheduler
    pipe.inverse_scheduler = inverse_scheduler

    pipe.enable_model_cpu_offload()

    print("DEBUG: Scheduler timesteps:", len(pipe.scheduler.timesteps))
    print("DEBUG: Inverse scheduler timesteps:", len(pipe.inverse_scheduler.timesteps))

    new_change_objects = []

    for obj in change_attr_objects:
        old_object_region = run_sam_postprocess(
            run_sam(obj[1], image_source, models),
            H, W, config
        ).astype(np.bool_)[np.newaxis, ...]

        new_object = obj[0].split(" #")[0]
        base_object = new_object.split(" ")[-1]
        mask_prompt = f"a {base_object}"
        new_prompt = f"a {new_object}"

        # Explicit inversion step
        inversion_result = pipe.invert(
            image=img,
            prompt=mask_prompt,
            inpaint_strength=inpaint_strength,
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            num_inference_steps=num_inference_steps,
        )

        image_latents = inversion_result.latents
        print("DEBUG: Latents shape from inversion:", image_latents.shape)  # Should match scheduler steps

        # Forward generation using inverted latents
        image = pipe(
            prompt=new_prompt,
            mask_image=old_object_region,
            image_latents=image_latents,
            guidance_scale=guidance_scale,
            inpaint_strength=inpaint_strength,
            generator=torch.Generator(device="cuda").manual_seed(inv_seed),
            negative_prompt="",
            num_inference_steps=num_inference_steps,
        ).images[0]

        all_latents, _ = get_all_latents(np.array(image), models, inv_seed)
        new_change_objects.append([
            old_object_region[0],
            all_latents,
        ])

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
    # parser.add_argument("--json-file", type=str, default="data/data.json", help="Path to data.json")
    parser.add_argument("--name", type=str, default="temp", help="Name of the image")
    parser.add_argument("--input-dir", type=str, default="data/input_dir", help="Path to the input directory")
    parser.add_argument("--output-dir", type=str, default="data/output_dir", help="Path to the output directory")
    parser.add_argument("--mode", type=str, default="self_correction", help="Mode of the demo", choices=["self_correction", "image_editing"])
    parser.add_argument("--config", type=str, default="personal_config.ini", help="Path to the config file")
    parser.add_argument("--benchmark", type=bool, default=False, help="Perform the benchmark")
    parser.add_argument("--model_key", type=str, default="CompVis/stable-diffusion-v1-4", help="Model key for initial image generation")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for initial image generation")
    parser.add_argument("--seed", type=int, default=42, help="Seed for initial image generation")
    args = parser.parse_args()

    # Generate the initial image
    image = generate_image(args.model_key, args.prompt, args.seed)
    input_fname = args.name
    fname = os.path.join(args.input_dir, f"{input_fname}.png") # fname = data/input_dir/temp.png
    image.save(fname)

    save_dir = os.path.join(args.output_dir, args.name) # save_dir = data/output_dir/temp
    os.makedirs(save_dir, exist_ok=True)
    parse.img_dir = os.path.join(save_dir, "tmp_imgs") # parse.img_dir = data/output_dir/temp/tmp_imgs
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

    # Prepare the evaluator
    evaluator = Evaluator()
    prompts = [args.prompt]
    if args.benchmark:
        prompts = get_lmd_prompts()["lmd"]

    for idx, prompt in enumerate(prompts):
        # Reset random seeds
        default_seed = int(config.get("SLD", "default_seed"))
        torch.manual_seed(default_seed)
        np.random.seed(default_seed)
        random.seed(default_seed)

        # Create an output directory for the current image
        dirname = save_dir # dirname = data/output_dir/temp
        if args.benchmark:
            dirname = os.path.join(save_dir, f"{idx:03d}") # dirname = data/output_dir/temp/{idx}
            os.makedirs(dirname, exist_ok=True)
        output_fname = os.path.join(dirname, "initial_image.jpg") # output_fname = data/output_dir/temp/{idx}/initial_image.png
        shutil.copy(fname, output_fname)
        log_file = os.path.join(dirname, "log.txt")
        set_file_handler(log_file)

        # Check whether we need to do self-correction
        attr_threshold = float(config.get("eval", "attr_detection_threshold")) 
        prim_threshold = float(config.get("eval", "prim_detection_threshold"))
        nms_threshold = float(config.get("eval", "nms_threshold"))

        # Step 0: Evaluate the initial image
        prompt = prompt.strip().rstrip(".")
        if args.benchmark:
            eval_type, eval_success = eval_prompt(prompt, fname, evaluator, 
                                                prim_score_threshold=prim_threshold, attr_score_threshold=attr_threshold, nms_threshold=nms_threshold, 
                                                use_class_aware_nms=True, use_cuda=True, verbose=False)
            if int(eval_success) >= 1:
                logging.info(f"Image {fname} is already correct!")
                continue

        chatgpt_data = {
            'llm_parser': None,
            'llm_controller': []
        }

        logging.info("-" * 5 + f" [Self-Correcting {fname}] " + "-" * 5)
        logging.info(f"Target Textual Prompt: {prompt}")

        # Step 1: Spot Objects with LLM
        logging.info("-" * 5 + f" Parsing Prompts " + "-" * 5)
        llm_parsed_prompt, spot_object_raw_response = run_llm_parser(prompt, config)
        entry = {"instructions": prompt, 
                 "output": [fname],
                "objects": llm_parsed_prompt["objects"], 
                "bg_prompt": llm_parsed_prompt["bg_prompt"],
                "neg_prompt": llm_parsed_prompt["neg_prompt"]
                }
        logging.info(f"* Objects: {entry['objects']}")
        logging.info(f"* Background: {entry['bg_prompt']}")
        logging.info(f"* Negation: {entry['neg_prompt']}")
        chatgpt_data["llm_parser"] = (prompt, spot_object_raw_response)

        num_round = int(config.get("SLD", "num_rounds", fallback=1))
        for i in range(num_round):
            logging.info(f"Round {i + 1}")

            # Step 2: Run open vocabulary detector
            logging.info("-" * 5 + f" Running Detector " + "-" * 5)
            attr_threshold = float(config.get("SLD", "attr_detection_threshold")) 
            prim_threshold = float(config.get("SLD", "prim_detection_threshold"))
            nms_threshold = float(config.get("SLD", "nms_threshold"))

            det_results = det.run(prompt, entry["objects"], entry["output"][-1],
                                attr_detection_threshold=attr_threshold, 
                                prim_detection_threshold=prim_threshold, 
                                nms_threshold=nms_threshold)
            
            # Step 3: Spot difference between detected results and initial prompts
            logging.info("-" * 5 + f" Getting Modification Suggestions " + "-" * 5)
            llm_suggestions, spot_difference_raw_response = run_llm_controller(prompt, det_results, config)
            entry["det_results"] = copy.deepcopy(det_results)
            entry["llm_suggestion"] = copy.deepcopy(llm_suggestions)
            logging.info(f"* Detection Restuls: {det_results}")
            logging.info(f"* LLM Suggestions: {llm_suggestions}")
            chatgpt_data["llm_controller"].append((prompt, spot_difference_raw_response))

            # Step 4: Check which objects to preserve, delete, add, reposition, or modify
            logging.info("-" * 5 + f" Editing Operations " + "-" * 5)
            (
                preserve_objs,
                deletion_objs,
                addition_objs,
                repositioning_objs,
                attr_modification_objs,
            ) = det.parse_list(det_results, llm_suggestions)
            logging.info(f"* Preservation: {preserve_objs}")
            logging.info(f"* Addition: {addition_objs}")
            logging.info(f"* Deletion: {deletion_objs}")
            logging.info(f"* Repositioning: {repositioning_objs}")
            logging.info(f"* Attribute Modification: {attr_modification_objs}")

            # Visualize the detection results
            parse.show_boxes(
                gen_boxes=entry["det_results"],
                additional_boxes=entry["llm_suggestion"],
                img=np.array(Image.open(entry["output"][-1])).astype(np.uint8),
                fname=os.path.join(dirname, f"det_result{i+1}.jpg"),
            )
            
            # Check if there are any changes to apply
            total_ops = len(deletion_objs) + len(addition_objs) + len(repositioning_objs) + len(attr_modification_objs)
            if (total_ops == 0):
                logging.info("-" * 5 + f" Results " + "-" * 5)
                final_output_fname = os.path.join(dirname, f"round{i+1}.jpg")
                shutil.copy(entry["output"][-1], final_output_fname)
                entry["output"].append(final_output_fname)
                logging.info("* No changes to apply!")
                logging.info(f"* Output File: {final_output_fname}")
                continue

            # Step 5: T2I Ops: Addition / Deletion / Repositioning / Attr. Modification
            logging.info("-" * 5 + f" Image Manipulation " + "-" * 5)

            deletion_region = get_remove_region(
                entry, deletion_objs, repositioning_objs, preserve_objs, models, config
            )
            repositioning_objs = get_repos_info(
                entry, repositioning_objs, models, config
            )
            new_attr_modification_objs = get_attrmod_latent(
                entry, attr_modification_objs, models, config
            )
            ret_dict = correction(
                entry, addition_objs, repositioning_objs,
                deletion_region, new_attr_modification_objs, 
                models, config
            )

            # Step 6: Save an intermediate file without the SDXL refinement
            logging.info("-" * 5 + f" Results " + "-" * 5)
            intermediate_output_fname = os.path.join(dirname, f"round{i+1}.jpg")
            Image.fromarray(ret_dict.image).save(intermediate_output_fname)
            entry["output"].append(intermediate_output_fname)
            logging.info(f"* Output File: {intermediate_output_fname}")
            utils.free_memory()

            # Evaluate again after self-correction!
            if args.benchmark:
                eval_type, eval_success = eval_prompt(prompt, intermediate_output_fname, evaluator, 
                                                    prim_score_threshold=prim_threshold, attr_score_threshold=attr_threshold, nms_threshold=nms_threshold, 
                                                    use_class_aware_nms=True, use_cuda=True, verbose=False)
                if int(eval_success) >= 1:
                    logging.info(f"Image {fname} is already correct!")
                else:
                    logging.info(f"Image {fname} is still incorrect!")

        sdxl_output_fname = os.path.join(dirname, f"final_{input_fname}.png")
        sdxl_refine(prompt, intermediate_output_fname, sdxl_output_fname)
        logging.info(f"* Output File (After SDXL): {sdxl_output_fname}")
        
        with open(os.path.join(dirname, "chatgpt_data.json"), 'w') as f:
            json.dump(chatgpt_data, f)
