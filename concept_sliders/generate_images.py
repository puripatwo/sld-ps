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
from tqdm.auto import tqdm

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from diffusers.pipelines import StableDiffusionXLPipeline

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import load_file

from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV


def flush():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='generateImages', description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='Name of the model.', type=str, required=True)
    parser.add_argument('--prompts_path', help='Path to .csv file with prompts.', type=str, required=True)
    parser.add_argument('--negative_prompts', help='Negative prompts.', type=str, required=False, default=None)
    parser.add_argument('--save_path', help='Folder of where to save the images.', type=str, required=True)
    parser.add_argument('--base', help='Version of stable diffusion to use.', type=str, required=False, default='1.4')
    parser.add_argument('--guidance_scale', help='Guidance to run eval.', type=float, required=False, default=7.5)
    parser.add_argument('--image_size', help='Image size used to train.', type=int, required=False, default=512)
    parser.add_argument('--from_case', help='Continue generating from case_number.', type=int, required=False, default=0)
    parser.add_argument('--till_case', help='Continue generating until case_number.', type=int, required=False, default=1000000)
    parser.add_argument('--num_samples', help='Number of samples per prompt.', type=int, required=False, default=5)
    parser.add_argument('--ddim_steps', help='DDIM steps of inference used to train.', type=int, required=False, default=50)
    parser.add_argument('--rank', help='Rank of the LoRA.', type=int, required=False, default=4)
    parser.add_argument('--start_noise', help='What time stamp to flip to the edited model', type=int, required=False, default=750)
    args = parser.parse_args()

    model_name = args.model_name
    rank = args.rank
    prompts_path = args.prompts_path
    save_path = args.save_path
    device = args.device
    guidance_scale = args.guidance_scale
    image_size = args.image_size
    ddim_steps = args.ddim_steps
    num_samples= args.num_samples
    from_case = args.from_case
    till_case = args.till_case
    start_noise = args.start_noise
    base = args.base

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = "cpu"

    negative_prompts_path = args.negative_prompts
    if negative_prompts_path is not None:
        negative_prompt = ''
        with open(negative_prompts_path, 'r') as fp:
            vals = json.load(fp)
            for val in vals:
                negative_prompt+=val+' ,'
        print(f'Negative prompt is being used: {negative_prompt}')
    else:
        negative_prompt = None

    # 1. Create directories for each scale.
    weight_dtype = torch.float16
    scales = [-2, -1, 0, 1, 2]
    folder_path = f'{save_path}/{model_name}'
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path+f'/all', exist_ok=True)
    scales_str = []
    for scale in scales:
        scale_str = f'{scale}'
        scale_str = scale_str.replace('0.5','half')
        scales_str.append(scale_str)
        os.makedirs(folder_path+f'/{scale_str}', exist_ok=True)

    # 2. Load in the scheduler, tokenizer, and models.
    revision = None
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required
    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    alpha = 1
    train_method = 'xattn'
    n = model_name.split('/')[-2]
    if 'noxattn' in n:
        train_method = 'noxattn'
    if 'hspace' in n:
        train_method+='-hspace'
        scales = [-5, -2, -1, 0, 1, 2, 5]
    if 'last' in n:
        train_method+='-last'
        scales = [-5, -2, -1, 0, 1, 2, 5]
    network_type = "c3lier"
    if train_method == 'xattn':
        network_type = 'lierla'

    modules = DEFAULT_TARGET_REPLACE
    if network_type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    # 3. Retrive trained LoRA adaptors.
    lora_weight = model_name
    network = LoRANetwork(
            unet,
            rank=rank,
            multiplier=1.0,
            alpha=alpha,
            train_method=train_method,
        ).to(device, dtype=weight_dtype)
    # network.load_state_dict(torch.load(lora_weight))
    state_dict = load_file(lora_weight)
    network.load_state_dict(state_dict)

    # 4. Convert the prompts .csv file to a DataFrame.
    df = pd.read_csv(prompts_path)
    prompts = df.prompt
    seeds = df.evaluation_seed
    case_numbers = df.case_number

    height = image_size
    width = image_size
    num_inference_steps = ddim_steps
    torch_device = device

    # 5. Loop through each prompt.
    for _, row in df.iterrows():
        print(str(row.prompt),str(row.evaluation_seed))

        prompt = [str(row.prompt)]*num_samples
        batch_size = len(prompt)
        seed = row.evaluation_seed

        case_number = row.case_number
        if not (case_number>=from_case and case_number<=till_case):
            continue

        # 6. Loop through each scale.
        images_list = []
        for scale in scales:
            generator = torch.manual_seed(seed)

            # torch_device = device
            # negative_prompt = None
            # height = 512
            # width = 512
            # guidance_scale = 7.5
            
            # 6.1. Tokenize the prompt.
            text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
            max_length = text_input.input_ids.shape[-1]

            # 6.2. Prepare the unconditional embeddings.
            if negative_prompt is None:
                uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            else:
                uncond_input = tokenizer([negative_prompt] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
            uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

            # 6.3. Prepare timesteps and latent variables.
            latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator)
            latents = latents.to(torch_device)
            noise_scheduler.set_timesteps(ddim_steps)
            latents = latents * noise_scheduler.init_noise_sigma
            latents = latents.to(weight_dtype)

            # 6.4. Denoising loop.
            for t in tqdm(noise_scheduler.timesteps):
                # Prepare the text embeddings
                if t > start_noise:
                    network.set_lora_slider(scale=0)
                else:
                    network.set_lora_slider(scale=scale)

                # Prepare embeddings for classifier-free guidance
                concat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                concat_text_embeddings = concat_text_embeddings.to(weight_dtype)

                # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

                # Predict the noise residual
                with network:
                    with torch.no_grad():
                        noise_pred = unet(latent_model_input, t, encoder_hidden_states=concat_text_embeddings).sample

                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # Compute the previous noisy sample x_t -> x_t-1
                latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

            # 6.5. Scale and decode the image latents with vae.
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                image = vae.decode(latents).sample

            # 6.6. Image post-processing.
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            images_list.append(pil_images)

        # 7. Loop through each generated image and store them.
        for num in range(num_samples):
            fig, ax = plt.subplots(1, len(images_list), figsize=(4*(len(scales)),4))
            for i, a in enumerate(ax):
                images_list[i][num].save(f'{folder_path}/{scales_str[i]}/{case_number}_{num}.png')
                a.imshow(images_list[i][num])
                a.set_title(f"{scales[i]}",fontsize=15)
                a.axis('off')
            fig.savefig(f'{folder_path}/all/{case_number}_{num}.png',bbox_inches='tight')
            plt.close()

    # 8. Clear memory and empty cache.
    del unet, network
    unet = None
    network = None
    torch.cuda.empty_cache()
    flush()
