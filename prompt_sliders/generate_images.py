import torch
import numpy as np
import matplotlib.image as mpimg
import copy
import gc
import argparse
import os, json, random, sys
import pandas as pd
import matplotlib.pyplot as plt
import glob, re
import warnings
warnings.filterwarnings("ignore")
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm

import safetensors.torch
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import DiffusionPipeline
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor

def flush():
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model.', type=str, required=True)
    parser.add_argument('--learned_embeds_path', help='Path to the learned embeddings.', type=str, required=True)
    parser.add_argument('--placeholder_token', help='The placeholder token to be used.', type=str, default="iid-1")
    parser.add_argument('--save_path', help='Folder of where to save the images.', type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    learned_embeds_path = args.learned_embeds_path
    placeholder_token = args.placeholder_token
    save_path = args.save_path

    # 1. Create directories for each scale.
    flush()
    weight_dtype = torch.float32
    scales = [0, 0.5, 1.0, 1.5]
    for scale in scales:
        os.makedirs(os.path.join(save_path, os.path.basename(model_name), str(scale)), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = "cpu"

    # 2. Load in the scheduler, tokenizer, and models.
    revision = None
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

    # 3. Check if the token already exists in the vocabulary.
    placeholder_token = f"<{placeholder_token}>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    # 4. Inject the learned embeddings in the position of the placeholder token id.
    loaded_embeds = safetensors.torch.load_file(learned_embeds_path)
    key = list(loaded_embeds.keys())[0]
    new_token_embed = loaded_embeds[key]
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight.data[placeholder_token_id] = new_token_embed.clone()

    unet.requires_grad_(False)
    unet.to(device, dtype=weight_dtype)
    vae.requires_grad_(False)
    vae.to(device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)
    text_encoder.to(device, dtype=weight_dtype)

    prompts = [
        f"a photo of a man, {placeholder_token}",
        ]
    start_noise = 800
    num_images_per_prompt = 1
    torch_device = device
    negative_prompt = None
    batch_size = 1
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5

    # 5. Loop through each prompt.
    for case_number, prompt in enumerate(prompts):
        for num in range(num_images_per_prompt):
            seed = random.randint(0, 5000)
            print(prompt, seed)

            # 6. Loop through each scale.
            for scale in scales:
                generator = torch.manual_seed(seed)

                # 6.1. Tokenize the prompt.
                text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                max_length = text_input.input_ids.shape[-1]
                batch_indices = torch.arange(len(text_input.input_ids))
                idx = text_input.input_ids.argmax(-1)

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
                    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
                    if t > start_noise and scale > 0.0:
                        text_embeddings[batch_indices, idx, :] = 0.0 * text_embeddings[batch_indices, idx, :]
                    else:
                        text_embeddings[batch_indices, idx, :] = scale * text_embeddings[batch_indices, idx, :]
                    
                    # Prepare embeddings for classifier-free guidance
                    concat_text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                    concat_text_embeddings = concat_text_embeddings.to(weight_dtype)

                    # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = noise_scheduler.scale_model_input(latent_model_input, timestep=t)

                    # Predict the noise residual
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

                # 7. Loop through each generated image and store them.
                for im in pil_images:
                    image_filename = f"{case_number}_{num}.png"
                    im.save(os.path.join(os.path.join(save_path, os.path.basename(model_name), str(scale)), image_filename))

    # 8. Clear memory and empty cache.
    del unet
    unet = None
    torch.cuda.empty_cache()
    flush()
