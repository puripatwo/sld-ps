import numpy as np
import torch

from . import pipelines
from . import sam
from . import models
from .models import model_dict
from .pipelines import DEFAULT_OVERALL_NEGATIVE_PROMPT, DEFAULT_SO_NEGATIVE_PROMPT 

import utils
from utils import parse, torch_device

vae, tokenizer, text_encoder, unet, scheduler, dtype = (
    model_dict.vae,
    model_dict.tokenizer,
    model_dict.text_encoder,
    model_dict.unet,
    model_dict.scheduler,
    model_dict.dtype,
)
version = "sld"

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion

H, W = height // 8, width // 8  # size of the latent
overall_batch_size = 1 # batch size: set to 1

guidance_scale = 2.5  # Scale for classifier-free guidance
guidance_attn_keys = pipelines.DEFAULT_GUIDANCE_ATTN_KEYS # attn keys for semantic guidance
offload_cross_attn_to_cpu = False

discourage_mask_below_confidence = 0.85 # discourage masks with confidence below
discourage_mask_below_coarse_iou = 0.25 # discourage masks with iou (with coarse binarized attention mask) below


def run(
    spec,
    bg_all_latents=None,
    bg_seed=1,
    overall_prompt_override="",
    fg_seed_start=20,
    frozen_step_ratio=0.5,
    num_inference_steps=50,
    loss_scale=5,
    loss_threshold=5.0,
    max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    max_index_step=0,
    overall_loss_scale=5,
    overall_loss_threshold=5.0,
    overall_max_iter=[4] * 5 + [3] * 5 + [2] * 5 + [2] * 5 + [1] * 10,
    overall_max_index_step=30,
    so_gligen_scheduled_sampling_beta=0.4,
    overall_gligen_scheduled_sampling_beta=0.4,
    overall_fg_top_p=0.2,
    overall_bg_top_p=0.2,
    overall_fg_weight=1.0,
    overall_bg_weight=4.0,
    ref_ca_loss_weight=2.0,
    so_center_box=False,
    fg_blending_ratio=0.1,
    so_negative_prompt=DEFAULT_SO_NEGATIVE_PROMPT,
    overall_negative_prompt=DEFAULT_OVERALL_NEGATIVE_PROMPT,
    so_horizontal_center_only=True,
    align_with_overall_bboxes=False,
    horizontal_shift_only=True,
    use_fast_schedule=False,
    # Transfer the cross-attention from single object generation (with ref_ca_saved_attns)
    # Use reference cross attention to guide the cross attention in the overall generation
    use_ref_ca=True,
    use_autocast=True,
    verbose=False,
):
    """
    spec: the spec for generation (see generate.py for how to construct a spec)
    bg_seed: background seed
    overall_prompt_override: use custom overall prompt (rather than the object prompt)
    fg_seed_start: each foreground has a seed (fg_seed_start + i), where i is the index of the foreground
    frozen_step_ratio: how many steps should be frozen (as a ratio to inference steps)
    num_inference_steps: number of inference steps
    (overall_)loss_scale: loss scale for per box or overall generation
    (overall_)loss_threshold: loss threshold for per box or overall generation, below which the loss will not be optimized to prevent artifacts
    (overall_)max_iter: max iterations of loss optimization for each step. If scaler, this is applied to all steps.
    (overall_)max_index_step: max index to apply loss optimization to.
    so_gligen_scheduled_sampling_beta and overall_gligen_scheduled_sampling_beta: the guidance steps with GLIGEN
    overall_fg_top_p and overall_bg_top_p: the top P fraction to optimize
    overall_fg_weight and overall_bg_weight: the weight for foreground and background optimization.
    ref_ca_loss_weight: weight for attention transfer (i.e., attention reference loss) to ensure the per-box generation is similar to overall generation in the masked region
    so_center_box: using centered box in single object generation to ensure better spatial control in the generation
    fg_blending_ratio: how much should each foreground initial noise deviate from the background initial noise (and each other)
    so_negative_prompt and overall_negative_prompt: negative prompt for single object (per-box) or overall generation
    so_horizontal_center_only: move to the center horizontally only
    align_with_overall_bboxes: Align the center of the mask, latents, and cross-attention with the center of the box in overall bboxes
    horizontal_shift_only: only shift horizontally for the alignment of mask, latents, and cross-attention
    use_fast_schedule: since the per-box generation, after the steps for latent and attention transfer, is only used by SAM (which does not need to be precise), we skip steps after the steps needed for transfer with a fast schedule.
    use_ref_ca: Use reference cross attention to guide the cross attention in the overall generation
    use_autocast: enable automatic mixed precision (saves memory and makes generation faster)
    Note: attention guidance is disabled for per-box generation by default (`max_index_step` set to 0) because we did not find it improving the results. Attention guidance and reference attention are still enabled for final guidance (overall generation). They greatly improve attribute binding compared to GLIGEN.
    """

    