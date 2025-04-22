import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import argparse
import os, json, random, sys
import glob, re
import copy
import gc
import inspect
import random
import abc
import shutil
import warnings
warnings.filterwarnings("ignore")
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from PIL import Image
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
import safetensors.torch
from safetensors.torch import load_file
from torch.optim.adam import Adam
import torch.nn.functional as nnf

import diffusers
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, UNet2DConditionModel, LMSDiscreteScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor, AttentionProcessor
from diffusers.pipelines import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput

import trainscripts.textsliders.ptp_utils as ptp_utils

LOW_RESOURCE = False
NUM_DDIM_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def load_512(image_path, left=0, right=0, top=0, bottom=0):
    # Load the image
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape

    # Clamp the crop values
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)

    # Crop the edges
    image = image[top:h-bottom, left:w-right]

    # Center the crop to a square image
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]

    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image


class NullInversion:
    def __init__(self, model):
        self.model = model
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(NUM_DDIM_STEPS)
        self.prompt = None
        self.context = None
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

    @property
    def scheduler(self):
        return self.model.scheduler

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt
    
    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device).to(self.model.vae.dtype)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents
    
    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        latents = latents.to(self.model.vae.dtype)
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).to(torch.float16).numpy()[0]
            image = (image * 255).astype(np.uint8)
        return image
    
    def prev_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample
    
    def next_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        latents_input = torch.cat([latents] * 2)
        latents_input = latents_input.to(self.model.unet.dtype)
        if context is None:
            context = self.context
        guidance_scale = 1 if is_forward else GUIDANCE_SCALE
        noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.model.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred
    
    @torch.no_grad()
    def ddim_loop(self, latent):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        latent = latent.to(self.model.unet.dtype)
        for i in range(NUM_DDIM_STEPS):
            t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            noise_pred = self.get_noise_pred_single(latent, t, cond_embeddings)
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.no_grad()
    def ddim_inversion(self, image):
        latent = self.image2latent(image)
        image_rec = self.latent2image(latent)
        ddim_latents = self.ddim_loop(latent)

        return image_rec, ddim_latents
    
    def null_optimization(self, latents, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = self.context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]

        bar = tqdm(total=num_inner_steps * NUM_DDIM_STEPS)
        for i in range(NUM_DDIM_STEPS):
            # 1. Clone and make unconditional embeddings trainable.
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True

            # 2. Create the optimizer.
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1. - i / 100.))

            # 3. Get the latent from the previous step.
            latent_prev = latents[len(latents) - i - 2]

            # 4. Predict noise with conditional embeddings.
            t = self.model.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)

            # 5. Optimize unconditional embeddings.
            for j in range(num_inner_steps):
                # 5.1. Predict unconditional noise and combine
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + GUIDANCE_SCALE * (noise_pred_cond - noise_pred_uncond)

                # 5.2. Perform the reverse step.
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)

                # 5.3. Compute the loss and optimize.
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()

                # 5.4. Perform early stopping.
                if loss_item < epsilon + i * 2e-5:
                    break

            for j in range(j + 1, num_inner_steps):
                bar.update()

            # 6. Update the latent for the next step.
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, False, context)
      
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, offsets=(0,0,0,0), num_inner_steps=10, early_stop_epsilon=1e-5, verbose=False):
        self.init_prompt(prompt)
        ptp_utils.register_attention_control(self.model, None)
        image_gt = load_512(image_path, *offsets)

        print("DDIM inversion...")
        image_rec, ddim_latents = self.ddim_inversion(image_gt)

        print("Null-text optimization...")
        uncond_embeddings = self.null_optimization(ddim_latents, num_inner_steps, early_stop_epsilon)

        return (image_gt, image_rec), ddim_latents[-1], uncond_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='Name of the model.', type=str, required=True)
    parser.add_argument('--image_path', help='Path to the image to be edited.', type=str, required=True)
    parser.add_argument('--image_prompt', help='Prompt of the image to be edited.', type=str, required=True)
    parser.add_argument('--learned_embeds_path', help='Path to the learned embeddings.', type=str, required=True)
    parser.add_argument('--placeholder_token', help='The placeholder token to be used.', type=str, default="iid-1")
    parser.add_argument('--save_path', help='Folder of where to save the images.', type=str, required=True)
    args = parser.parse_args()

    model_name = args.model_name
    image_path = args.image_path
    image_prompt = f"{args.image_prompt}, <{args.placeholder_token}>"
    learned_embeds_path = args.learned_embeds_path
    placeholder_token = args.placeholder_token
    save_path = args.save_path

    # 1. Create directories to store the results.
    flush()
    weight_dtype = torch.float32
    scales = [0, 0.5, 1.0, 1.5]
    os.makedirs(os.path.join(save_path, os.path.basename(model_name), os.path.basename(image_path)), exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:0")
    else:
        device = "cpu"

    # 2. Prepare for Null Inversion.
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, torch_dtype=weight_dtype).to(device)
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")
    tokenizer = ldm_stable.tokenizer

    # 3. Perform Null Inversion.
    null_inversion = NullInversion(ldm_stable)
    (image_gt, image_enc), x_t, uncond_embeddings = null_inversion.invert(image_path, image_prompt, offsets=(0, 0, 0, 0), verbose=True)
    Image.fromarray(image_enc)
    uncond_embeddings_copy = copy.deepcopy(uncond_embeddings)
    del ldm_stable
    flush()

    # 4. Load in the scheduler, tokenizer, and models.
    revision = None
    pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    noise_scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", revision=revision)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision=revision)
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision=revision)
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision=revision)

    # 5. Check if the token already exists in the vocabulary.
    placeholder_token = f"<{placeholder_token}>"
    if placeholder_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([placeholder_token])
        placeholder_token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        text_encoder.resize_token_embeddings(len(tokenizer))
    
    # 6. Inject the learned embeddings in the position of the placeholder token id.
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

    start_noise = 800
    num_images_per_prompt = 1
    torch_device = device
    negative_prompt = None
    batch_size = 1
    height = 512
    width = 512
    ddim_steps = 50
    guidance_scale = 7.5

    # 7. Loop through each scale.
    for scale in scales:
        # 7.1. Tokenize the prompt.
        text_input = tokenizer(image_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        max_length = text_input.input_ids.shape[-1]
        batch_indices = torch.arange(len(text_input.input_ids))
        idx = text_input.input_ids.argmax(-1)

        # 7.2. Prepare timesteps and latent variables.
        noise_scheduler.set_timesteps(ddim_steps)
        latents = x_t * noise_scheduler.init_noise_sigma
        latents = latents.to(unet.dtype)

        # 7.3. Denoising loop.
        cnt = -1
        for t in tqdm(noise_scheduler.timesteps):
            cnt += 1

            # Prepare the text embeddings
            text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
            if t > start_noise and scale > 0.0:
                text_embeddings[batch_indices, idx, :] = 0.0 * text_embeddings[batch_indices, idx, :]
            else:
                text_embeddings[batch_indices, idx, :] = scale * text_embeddings[batch_indices, idx, :]
            
            # Prepare embeddings for classifier-free guidance
            concat_text_embeddings = torch.cat([uncond_embeddings_copy[cnt].expand(*text_embeddings.shape), text_embeddings])
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

        # 7.5. Scale and decode the image latents with vae.
        latents = 1 / 0.18215 * latents
        with torch.no_grad():
            image = vae.decode(latents).sample

        # 7.6. Image post-processing.
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (image * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]

        # 7.7. Loop through each generated image and store them.
        for im in pil_images:
            image_filename = f"{scale}.png"
            im.save(os.path.join(os.path.join(save_path, os.path.basename(model_name), os.path.basename(image_path)), image_filename))
