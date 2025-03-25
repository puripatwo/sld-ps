import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import argparse
import os, json, random
import glob, re
import copy
import gc
import inspect
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm

import diffusers
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import load_file

from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
# import trainscripts.textsliders.ptp_utils as ptp_utils

LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def flush():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='editImages', description='Edit Images using Diffusers Code')
    parser.add_argument('--model_name', help='Name of the model.', type=str, required=True)
    parser.add_argument('--image_path', help='Path to the image to be edited.', type=str, required=True)
    parser.add_argument('--image_prompt', help='Prompt of the image to be edited.', type=str, required=True)
    parser.add_argument('--save_path', help='Folder of where to save the images.', type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    image_path = args.image_path
    image_prompt = args.image_prompt
    save_path = args.save_path

    # 1
    weight_dtype = torch.float32
    device = torch.device("cuda")

    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, torch_dtype=weight_dtype).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    # 2
    # null_inversion = NullInversion(ldm_stable)
    # (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, image_prompt, offsets=(0, 0, 0, 0), verbose=True)
    # Image.fromarray(image_enc)
    # uncond_embeddings_copy = copy.deepcopy(uncond_embeddings)
    # del ldm_stable
    # flush()

    # 3
    width = 512
    height = 512
    steps = 50
    cfg_scale = 7.5
    revision = None
    rank = 4

    # 4
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    noise_scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)
    
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)

    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)

    # 5
    alpha = 1

    if 'full' in model_name:
        train_method = 'full'
    elif 'noxattn' in model_name:
        train_method = 'noxattn'
    elif 'xattn' in model_name:
        train_method = 'xattn'
    else:
        train_method = 'noxattn'

    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    # 6
    lora_weight = model_name
    network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(device, dtype=weight_dtype)
    network.load_state_dict(torch.load(lora_weight))

    negative_prompt = None
    batch_size = 1
    ddim_steps = 50
    guidance_scale = 7.5
    start_noise = 500
    