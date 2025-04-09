import torch
from typing import Optional
from tqdm import tqdm

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import UNet2DConditionModel, SchedulerMixin
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import randn_tensor

from model_util import SDXL_TEXT_ENCODER_TYPE


# https://www.crosslabs.org/blog/diffusion-with-offset-noise
def apply_noise_offset(latents: torch.FloatTensor, noise_offset: float):
    latents = latents + noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
    return latents


def get_optimizer(name: str):
    name = name.lower()

    if name.startswith("dadapt"):
        import dadaptation

        if name == "dadaptadam":
            return dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            return dadaptation.DAdaptLion
        else:
            raise ValueError("DAdapt optimizer must be dadaptadam or dadaptlion")

    elif name.endswith("8bit"):  # 検証してない
        import bitsandbytes as bnb

        if name == "adam8bit":
            return bnb.optim.Adam8bit
        elif name == "lion8bit":
            return bnb.optim.Lion8bit
        else:
            raise ValueError("8bit optimizer must be adam8bit or lion8bit")

    else:
        if name == "adam":
            return torch.optim.Adam
        elif name == "adamw":
            return torch.optim.AdamW
        elif name == "lion":
            from lion_pytorch import Lion

            return Lion
        elif name == "prodigy":
            import prodigyopt
            
            return prodigyopt.Prodigy
        else:
            raise ValueError("Optimizer must be adam, adamw, lion or Prodigy")


def get_lr_scheduler(
    name: Optional[str],
    optimizer: torch.optim.Optimizer,
    max_iterations: Optional[int],
    lr_min: Optional[float],
    **kwargs,
):
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_iterations, eta_min=lr_min, **kwargs
        )
    elif name == "cosine_with_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max_iterations // 10, T_mult=2, eta_min=lr_min, **kwargs
        )
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max_iterations // 100, gamma=0.999, **kwargs
        )
    elif name == "constant":
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, **kwargs)
    elif name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, factor=0.5, total_iters=max_iterations // 100, **kwargs
        )
    else:
        raise ValueError(
            "Scheduler must be cosine, cosine_with_restarts, step, linear or constant"
        )
# -----------------------------------------------------------


def text_tokenize(
    tokenizer: CLIPTokenizer,  # 普通ならひとつ、XLならふたつ！
    prompts: list[str],
):
    return tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids


def text_encode(text_encoder: CLIPTextModel, tokens):
    return text_encoder(tokens.to(text_encoder.device))[0]


def encode_prompts(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTokenizer,
    prompts: list[str],
):

    text_tokens = text_tokenize(tokenizer, prompts)
    text_embeddings = text_encode(text_encoder, text_tokens)

    return text_embeddings
# -----------------------------------------------------------

UNET_IN_CHANNELS = 4  # Stable Diffusion の in_channels は 4 で固定。XLも同じ。
VAE_SCALE_FACTOR = 8  # 2 ** (len(vae.config.block_out_channels) - 1) = 8


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width


@torch.no_grad()
def get_noisy_image(
    img,
    vae,
    generator,
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    total_timesteps: int = 1000,
    start_timesteps=0,
    
    **kwargs,
):
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    image = img
    im_orig = image
    device = vae.device
    image = image_processor.preprocess(image).to(device)

    init_latents = vae.encode(image).latent_dist.sample(None)
    init_latents = vae.config.scaling_factor * init_latents
    init_latents = torch.cat([init_latents], dim=0)

    shape = init_latents.shape
    noise = randn_tensor(shape, generator=generator, device=device)

    time_ = total_timesteps
    timestep = scheduler.timesteps[time_:time_+1]
    init_latents = scheduler.add_noise(init_latents, noise, timestep)

    return init_latents, noise
# -----------------------------------------------------------
 

# ref: https://github.com/huggingface/diffusers/blob/0bab447670f47c28df60fbd2f6a0f833f75a16f5/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L721
def predict_noise(
    unet: UNet2DConditionModel,
    scheduler: SchedulerMixin,
    timestep: int,  # 現在のタイムステップ
    latents: torch.FloatTensor,
    text_embeddings: torch.FloatTensor,  # uncond な text embed と cond な text embed を結合したもの
    guidance_scale=7.5,
) -> torch.FloatTensor:
    
    # Expand the latents if we are doing classifier-free guidance to avoid doing two forward passes
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

    # Predict the noise residual
    noise_pred = unet(
        latent_model_input,
        timestep,
        encoder_hidden_states=text_embeddings,
    ).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    guided_target = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    return guided_target


def concat_embeddings(
    unconditional: torch.FloatTensor,
    conditional: torch.FloatTensor,
    n_imgs: int,
):
    return torch.cat([unconditional, conditional]).repeat_interleave(n_imgs, dim=0)
