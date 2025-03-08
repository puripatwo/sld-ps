import numpy as np
import torch
import gc
import warnings

from PIL import Image
from tqdm import tqdm

import utils
from utils import torch_device
from .attention import GatedSelfAttentionDense
from .models import process_input_embeddings

DEFAULT_GUIDANCE_ATTN_KEYS = [
    ("mid", 0, 0, 0),
    ("up", 1, 0, 0),
    ("up", 1, 1, 0),
    ("up", 1, 2, 0),
]


# ----------------- LATENT BACKWARD GUIDANCE -----------------
@torch.no_grad()
def encode(model_dict, image, generator):
    vae, dtype = model_dict.vae, model_dict.dtype

    if isinstance(image, Image.Image):
        w, h = image.size
        assert (
            w % 8 == 0 and h % 8 == 0
        ), f"h ({h}) and w ({w}) should be a multiple of 8"
        # w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        # image = np.array(image.resize((w, h), resample=Image.Resampling.LANCZOS))[None, :]
        image = np.array(image)

    if isinstance(image, np.ndarray):
        assert (
            image.dtype == np.uint8
        ), f"Should have dtype uint8 (dtype: {image.dtype})"
        image = image.astype(np.float32) / 255.0
        image = image[None, ...]
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)

    assert isinstance(image, torch.Tensor), f"type of image: {type(image)}"

    # Encode the image
    image = image.to(device=torch_device, dtype=dtype)
    latents = vae.encode(image).latent_dist.sample(generator)
    latents = vae.config.scaling_factor * latents

    return latents


def get_inverse_timesteps(inverse_scheduler, num_inference_steps, strength):
    # get the original timestep using init_timestep
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

    t_start = max(num_inference_steps - init_timestep, 0)

    # safety for t_start overflow to prevent empty timsteps slice
    if t_start == 0:
        return inverse_scheduler.timesteps, num_inference_steps
    timesteps = inverse_scheduler.timesteps[:-t_start]

    return timesteps, num_inference_steps - t_start


@torch.no_grad()
def invert(model_dict, latents, input_embeddings, num_inference_steps, guidance_scale=7.5):
    """
    latents: encoded from the image, should not have noise (t = 0)

    returns inverted_latents for all time steps
    """
    vae, tokenizer, text_encoder, unet, scheduler, inverse_scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.inverse_scheduler,
        model_dict.dtype,
    )
    text_embeddings, uncond_embeddings, cond_embeddings = input_embeddings

    # Set the number of inference steps
    inverse_scheduler.set_timesteps(num_inference_steps, device=latents.device)
    timesteps, num_inference_steps = get_inverse_timesteps(inverse_scheduler, num_inference_steps, strength=1.0)
    inverted_latents = [latents.cpu()]

    # Compute the latents for all timesteps
    for t in tqdm(timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        if guidance_scale > 0.0:
            latent_model_input = torch.cat([latents] * 2)

            latent_model_input = inverse_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )
        else:
            latent_model_input = latents

            latent_model_input = inverse_scheduler.scale_model_input(
                latent_model_input, timestep=t
            )

            # predict the noise residual
            with torch.no_grad():
                noise_pred_uncond = unet(
                    latent_model_input, t, encoder_hidden_states=uncond_embeddings
                ).sample

            # perform guidance
            noise_pred = noise_pred_uncond

        # compute the previous noisy sample x_t -> x_t-1
        latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample

        inverted_latents.append(latents.cpu())

    assert len(inverted_latents) == len(timesteps) + 1
    inverted_latents = torch.stack(list(reversed(inverted_latents)), dim=0)

    return inverted_latents


DEFAULT_SO_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate, two, many, group, occlusion, occluded, side, border, collate"
DEFAULT_OVERALL_NEGATIVE_PROMPT = "artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, duplicate"


def get_all_latents(img_np, models, inv_seed=1):
    generator = torch.cuda.manual_seed(inv_seed)
    cln_latents = encode(models.model_dict, img_np, generator)
    # Magic prompt
    # Have tried using the parsed bg prompt from the LLM, but it doesn't work well
    prompt = "A realistic photo of a scene"
    input_embeddings = models.encode_prompts(
        prompts=[prompt],
        tokenizer=models.model_dict.tokenizer,
        text_encoder=models.model_dict.text_encoder,
        negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
        one_uncond_input_only=False,
    )
    # Get all hidden latents
    all_latents = invert(
        models.model_dict,
        cln_latents,
        input_embeddings,
        num_inference_steps=50,
        guidance_scale=2.5,
    )
    return all_latents, input_embeddings


# ----------------- GLIGEN -----------------
