# Apply self-correcting LLM-controlled diffusion (SLD)
# Reference: Wu et al., "Self-Correcting LLM-Controlled Diffusion Models"
# https://arxiv.org/abs/2309.16668

import numpy as np
import torch
import gc
import warnings

from PIL import Image
from tqdm import tqdm

import utils
from utils import schedule, guidance, torch_device
from utils.latents import get_scaled_latents
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


@torch.no_grad()
def decode(vae, latents):
    # scale and decode the image latents with vae
    scaled_latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(scaled_latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")

    return images


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
def latent_backward_guidance(
    scheduler,
    unet,
    cond_embeddings,
    index,
    bboxes,
    object_positions,
    t,
    latents,
    loss,
    loss_scale=30,
    loss_threshold=0.2,
    max_iter=5,
    max_index_step=10,
    cross_attention_kwargs=None,
    ref_ca_saved_attns=None,
    guidance_attn_keys=None,
    verbose=False,
    clear_cache=False,
    **kwargs,
):
    iteration = 0

    if index < max_index_step:
        if isinstance(max_iter, list):
            if len(max_iter) > index:
                max_iter = max_iter[index]
            else:
                max_iter = max_iter[-1]

        if verbose:
            print(
                f"time index {index}, loss: {loss.item()/loss_scale:.3f} (de-scaled with scale {loss_scale:.1f}), loss threshold: {loss_threshold:.3f}"
            )

        while (
            loss.item() / loss_scale > loss_threshold
            and iteration < max_iter
            and index < max_index_step
        ):
            saved_attn = {}
            full_cross_attention_kwargs = {
                "save_attn_to_dict": saved_attn,
                "save_keys": guidance_attn_keys,
            }

            if cross_attention_kwargs is not None:
                full_cross_attention_kwargs.update(cross_attention_kwargs)

            latents.requires_grad_(True)
            latent_model_input = latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            unet(
                latent_model_input,
                t,
                encoder_hidden_states=cond_embeddings,
                return_cross_attention_probs=False,
                cross_attention_kwargs=full_cross_attention_kwargs,
            )

            # TODO: could return the attention maps for the required blocks only and not necessarily the final output
            # update latents with guidance

            loss = (
                guidance.compute_ca_lossv3(
                    saved_attn=saved_attn,
                    bboxes=bboxes,
                    object_positions=object_positions,
                    guidance_attn_keys=guidance_attn_keys,
                    ref_ca_saved_attns=ref_ca_saved_attns,
                    index=index,
                    verbose=verbose,
                    **kwargs,
                )
                * loss_scale
            )

            if torch.isnan(loss):
                print("**Loss is NaN**")

            del full_cross_attention_kwargs, saved_attn
            # call gc.collect() here may release some memory

            grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents])[0]

            latents.requires_grad_(False)

            if hasattr(scheduler, "sigmas"):
                latents = latents - grad_cond * scheduler.sigmas[index] ** 2
            elif hasattr(scheduler, "alphas_cumprod"):
                warnings.warn("Using guidance scaled with alphas_cumprod")
                # Scaling with classifier guidance
                alpha_prod_t = scheduler.alphas_cumprod[t]
                # Classifier guidance: https://arxiv.org/pdf/2105.05233.pdf
                # DDIM: https://arxiv.org/pdf/2010.02502.pdf
                scale = (1 - alpha_prod_t) ** (0.5)
                latents = latents - scale * grad_cond
            else:
                # NOTE: no scaling is performed
                warnings.warn("No scaling in guidance is performed")
                latents = latents - grad_cond
            iteration += 1

            if clear_cache:
                utils.free_memory()

            if verbose:
                print(
                    f"time index {index}, loss: {loss.item()/loss_scale:.3f}, loss threshold: {loss_threshold:.3f}, iteration: {iteration}"
                )

    return latents, loss


def gligen_enable_fuser(unet, enabled=True):
    for module in unet.modules():
        if isinstance(module, GatedSelfAttentionDense):
            module.enabled = enabled


def prepare_gligen_condition(
    bboxes, phrases, dtype, tokenizer, text_encoder, num_images_per_prompt
):
    batch_size = len(bboxes)

    assert len(phrases) == len(bboxes)
    max_objs = 30

    n_objs = min(max([len(bboxes_item) for bboxes_item in bboxes]), max_objs)
    boxes = torch.zeros((batch_size, max_objs, 4), device=torch_device, dtype=dtype)
    phrase_embeddings = torch.zeros(
        (batch_size, max_objs, 768), device=torch_device, dtype=dtype
    )
    # masks is a 1D tensor deciding which of the enteries to be enabled
    masks = torch.zeros((batch_size, max_objs), device=torch_device, dtype=dtype)

    if n_objs > 0:
        for idx, (bboxes_item, phrases_item) in enumerate(zip(bboxes, phrases)):
            # the length of `bboxes_item` could be smaller than `n_objs` because n_objs takes the max of item length
            bboxes_item = torch.tensor(bboxes_item[:n_objs])
            boxes[idx, : bboxes_item.shape[0]] = bboxes_item

            tokenizer_inputs = tokenizer(
                phrases_item[:n_objs], padding=True, return_tensors="pt"
            ).to(torch_device)
            _phrase_embeddings = text_encoder(**tokenizer_inputs).pooler_output
            phrase_embeddings[idx, : _phrase_embeddings.shape[0]] = _phrase_embeddings
            assert (
                bboxes_item.shape[0] == _phrase_embeddings.shape[0]
            ), f"{bboxes_item.shape[0]} != {_phrase_embeddings.shape[0]}"

            masks[idx, : bboxes_item.shape[0]] = 1

    # Classifier-free guidance
    repeat_times = num_images_per_prompt * 2
    condition_len = batch_size * repeat_times

    boxes = boxes.repeat(repeat_times, 1, 1)
    phrase_embeddings = phrase_embeddings.repeat(repeat_times, 1, 1)
    masks = masks.repeat(repeat_times, 1)
    masks[: condition_len // 2] = 0

    # print("shapes:", boxes.shape, phrase_embeddings.shape, masks.shape)

    return boxes, phrase_embeddings, masks, condition_len


@torch.no_grad()
def generate_gligen(
    model_dict,
    latents,
    input_embeddings,
    num_inference_steps,
    bboxes,
    phrases,
    num_images_per_prompt=1,
    gligen_scheduled_sampling_beta: float = 0.3,
    guidance_scale=7.5,
    frozen_steps=25,
    frozen_mask=None,
    return_saved_cross_attn=False,
    saved_cross_attn_keys=None,
    return_cond_ca_only=False,
    return_token_ca_only=None,
    offload_cross_attn_to_cpu=False,
    offload_latents_to_cpu=True,
    semantic_guidance=False,
    semantic_guidance_bboxes=None,
    semantic_guidance_object_positions=None,
    semantic_guidance_kwargs=None,
    return_box_vis=False,
    show_progress=True,
    save_all_latents=False,
    batched_condition=False,
    dynamic_num_inference_steps=False,
    fast_after_steps=None,
    fast_rate=2,
):
    """
    The `bboxes` should be a list, rather than a list of lists (one box per phrase, we can have multiple duplicated phrases).
    batched:
        Enabled: bboxes and phrases should be a list (batch dimension) of items (specify the bboxes/phrases of each image in the batch).
        Disabled: bboxes and phrases should be a list of bboxes and phrases specifying the bboxes/phrases of one image (no batch dimension).
    """
    vae, tokenizer, text_encoder, unet, scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.dtype,
    )

    text_embeddings, _, cond_embeddings = process_input_embeddings(input_embeddings)

    if latents.dim() == 5:
        # latents_all from the input side, different from the latents_all to be saved
        latents_all_input = latents
        latents = latents[0]
    else:
        latents_all_input = None

    # Just in case that we have in-place ops
    latents = latents.clone()

    if save_all_latents:
        # offload to cpu to save space
        if offload_latents_to_cpu:
            latents_all = [latents.cpu()]
        else:
            latents_all = [latents]

    scheduler.set_timesteps(num_inference_steps)
    if fast_after_steps is not None:
        scheduler.timesteps = schedule.get_fast_schedule(
            scheduler.timesteps, fast_after_steps, fast_rate
        )

    if dynamic_num_inference_steps:
        original_num_inference_steps = scheduler.num_inference_steps

    if frozen_mask is not None:
        frozen_mask = frozen_mask.to(dtype=dtype).clamp(0.0, 1.0)

    # 5.1 Prepare GLIGEN variables
    if not batched_condition:
        # Add batch dimension to bboxes and phrases
        bboxes, phrases = [bboxes], [phrases]

    boxes, phrase_embeddings, masks, condition_len = prepare_gligen_condition(
        bboxes, phrases, dtype, tokenizer, text_encoder, num_images_per_prompt
    )

    if semantic_guidance_bboxes and semantic_guidance:
        loss = torch.tensor(10000.0)
        # TODO: we can also save necessary tokens only to save memory.
        # offload_guidance_cross_attn_to_cpu does not save too much since we only store attention map for each timestep.
        guidance_cross_attention_kwargs = {
            "offload_cross_attn_to_cpu": False,
            "enable_flash_attn": False,
            "gligen": {
                "boxes": boxes[: condition_len // 2],
                "positive_embeddings": phrase_embeddings[: condition_len // 2],
                "masks": masks[: condition_len // 2],
                "fuser_attn_kwargs": {
                    "enable_flash_attn": False,
                },
            },
        }

    if return_saved_cross_attn:
        saved_attns = []

    main_cross_attention_kwargs = {
        "offload_cross_attn_to_cpu": offload_cross_attn_to_cpu,
        "return_cond_ca_only": return_cond_ca_only,
        "return_token_ca_only": return_token_ca_only,
        "save_keys": saved_cross_attn_keys,
        "gligen": {
            "boxes": boxes,
            "positive_embeddings": phrase_embeddings,
            "masks": masks,
        },
    }

    timesteps = scheduler.timesteps

    num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
    gligen_enable_fuser(unet, True)

    for index, t in enumerate(tqdm(timesteps, disable=not show_progress)):
        # Scheduled sampling
        if index == num_grounding_steps:
            gligen_enable_fuser(unet, False)

        if semantic_guidance_bboxes and semantic_guidance:
            with torch.enable_grad():
                latents, loss = latent_backward_guidance(
                    scheduler,
                    unet,
                    cond_embeddings,
                    index,
                    semantic_guidance_bboxes,
                    semantic_guidance_object_positions,
                    t,
                    latents,
                    loss,
                    cross_attention_kwargs=guidance_cross_attention_kwargs,
                    **semantic_guidance_kwargs,
                )
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        main_cross_attention_kwargs["save_attn_to_dict"] = {}

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=main_cross_attention_kwargs,
        ).sample

        if return_saved_cross_attn:
            saved_attns.append(main_cross_attention_kwargs["save_attn_to_dict"])

            del main_cross_attention_kwargs["save_attn_to_dict"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if dynamic_num_inference_steps:
            schedule.dynamically_adjust_inference_steps(scheduler, index, t)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if frozen_mask is not None and index < frozen_steps:
            latents = latents_all_input[index + 1] * frozen_mask + latents * (
                1.0 - frozen_mask
            )

        # Do not save the latents in the fast steps
        if save_all_latents and (fast_after_steps is None or index < fast_after_steps):
            if offload_latents_to_cpu:
                latents_all.append(latents.cpu())
            else:
                latents_all.append(latents)

    if dynamic_num_inference_steps:
        # Restore num_inference_steps to avoid confusion in the next generation if it is not dynamic
        scheduler.num_inference_steps = original_num_inference_steps

    # Turn off fuser for typical SD
    gligen_enable_fuser(unet, False)
    images = decode(vae, latents)

    ret = [latents, images]
    if return_saved_cross_attn:
        ret.append(saved_attns)
    if return_box_vis:
        pil_images = [
            utils.draw_box(Image.fromarray(image), bboxes_item, phrases_item)
            for image, bboxes_item, phrases_item in zip(images, bboxes, phrases)
        ]
        ret.append(pil_images)
    if save_all_latents:
        latents_all = torch.stack(latents_all, dim=0)
        ret.append(latents_all)

    return tuple(ret)


@torch.no_grad()
def generate_gligen_final(
    model_dict,
    latents,
    input_embeddings,
    num_inference_steps,
    bboxes,
    phrases,
    num_images_per_prompt=1,
    gligen_scheduled_sampling_beta: float = 0.3,
    guidance_scale=7.5,
    frozen_steps=20,
    frozen_mask=None,
    return_saved_cross_attn=False,
    saved_cross_attn_keys=None,
    return_cond_ca_only=False,
    return_token_ca_only=None,
    offload_cross_attn_to_cpu=False,
    offload_latents_to_cpu=True,
    semantic_guidance=False,
    semantic_guidance_bboxes=None,
    semantic_guidance_object_positions=None,
    semantic_guidance_kwargs=None,
    return_box_vis=False,
    show_progress=True,
    save_all_latents=False,
    batched_condition=False,
    dynamic_num_inference_steps=False,
    fast_after_steps=None,
    fast_rate=2,
    initial_bg=None,
):
    """
    The `bboxes` should be a list, rather than a list of lists (one box per phrase, we can have multiple duplicated phrases).
    batched:
        Enabled: bboxes and phrases should be a list (batch dimension) of items (specify the bboxes/phrases of each image in the batch).
        Disabled: bboxes and phrases should be a list of bboxes and phrases specifying the bboxes/phrases of one image (no batch dimension).
    """
    # 1. Process the input embeddings.
    vae, tokenizer, text_encoder, unet, scheduler, dtype = (
        model_dict.vae,
        model_dict.tokenizer,
        model_dict.text_encoder,
        model_dict.unet,
        model_dict.scheduler,
        model_dict.dtype,
    )

    text_embeddings, _, cond_embeddings = process_input_embeddings(input_embeddings)

    # 2. Initialize the latents.
    if latents.dim() == 5:
        # latents_all from the input side, different from the latents_all to be saved
        latents_all_input = latents
        latents = latents[0]
    else:
        latents_all_input = None

    # Just in case that we have in-place ops
    if initial_bg is None:
        generator = torch.manual_seed(
            6666
        )  # Seed generator to create the inital latent noise
        latents = get_scaled_latents(
            1, unet.config.in_channels, 512, 512, generator, dtype, scheduler
        )
    else:
        latents = initial_bg[0].cuda()

    # 3. Apply the frozen mask.
    latents = latents_all_input[0].cuda() * frozen_mask + latents * (1.0 - frozen_mask)

    latents = latents.clone()

    if save_all_latents:
        # offload to cpu to save space
        if offload_latents_to_cpu:
            latents_all = [latents.cpu()]
        else:
            latents_all = [latents]

    # 4. Set up the diffusion steps.
    scheduler.set_timesteps(num_inference_steps)
    if fast_after_steps is not None:
        scheduler.timesteps = schedule.get_fast_schedule(
            scheduler.timesteps, fast_after_steps, fast_rate
        )

    if dynamic_num_inference_steps:
        original_num_inference_steps = scheduler.num_inference_steps

    if frozen_mask is not None:
        frozen_mask = frozen_mask.to(dtype=dtype).clamp(0.0, 1.0)

    # 5. Prepare GLIGEN variables.
    if not batched_condition:
        # Add batch dimension to bboxes and phrases
        bboxes, phrases = [bboxes], [phrases]

    boxes, phrase_embeddings, masks, condition_len = prepare_gligen_condition(
        bboxes, phrases, dtype, tokenizer, text_encoder, num_images_per_prompt
    )

    # 6. Apply semantic guiidance.
    if semantic_guidance_bboxes and semantic_guidance:
        loss = torch.tensor(10000.0)
        # TODO: we can also save necessary tokens only to save memory.
        # offload_guidance_cross_attn_to_cpu does not save too much since we only store attention map for each timestep.
        guidance_cross_attention_kwargs = {
            "offload_cross_attn_to_cpu": False,
            "enable_flash_attn": False,
            "gligen": {
                "boxes": boxes[: condition_len // 2],
                "positive_embeddings": phrase_embeddings[: condition_len // 2],
                "masks": masks[: condition_len // 2],
                "fuser_attn_kwargs": {
                    "enable_flash_attn": False,
                },
            },
        }

    if return_saved_cross_attn:
        saved_attns = []

    main_cross_attention_kwargs = {
        "offload_cross_attn_to_cpu": offload_cross_attn_to_cpu,
        "return_cond_ca_only": return_cond_ca_only,
        "return_token_ca_only": return_token_ca_only,
        "save_keys": saved_cross_attn_keys,
        "gligen": {
            "boxes": boxes,
            "positive_embeddings": phrase_embeddings,
            "masks": masks,
        },
    }

    timesteps = scheduler.timesteps

    num_grounding_steps = int(gligen_scheduled_sampling_beta * len(timesteps))
    gligen_enable_fuser(unet, True)

    # 8. Perform the diffusion process.
    for index, t in enumerate(tqdm(timesteps, disable=not show_progress)):
        # Scheduled sampling
        if index == num_grounding_steps:
            gligen_enable_fuser(unet, False)

        if semantic_guidance_bboxes and semantic_guidance:
            with torch.enable_grad():
                latents, loss = latent_backward_guidance(
                    scheduler,
                    unet,
                    cond_embeddings,
                    index,
                    semantic_guidance_bboxes,
                    semantic_guidance_object_positions,
                    t,
                    latents,
                    loss,
                    cross_attention_kwargs=guidance_cross_attention_kwargs,
                    **semantic_guidance_kwargs,
                )
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

        main_cross_attention_kwargs["save_attn_to_dict"] = {}

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
            cross_attention_kwargs=main_cross_attention_kwargs,
        ).sample

        if return_saved_cross_attn:
            saved_attns.append(main_cross_attention_kwargs["save_attn_to_dict"])

            del main_cross_attention_kwargs["save_attn_to_dict"]

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        if dynamic_num_inference_steps:
            schedule.dynamically_adjust_inference_steps(scheduler, index, t)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

        if frozen_mask is not None and index < frozen_steps:
            latents = latents_all_input[index + 1] * frozen_mask + latents * (
                1.0 - frozen_mask
            )

        # Do not save the latents in the fast steps
        if save_all_latents and (fast_after_steps is None or index < fast_after_steps):
            if offload_latents_to_cpu:
                latents_all.append(latents.cpu())
            else:
                latents_all.append(latents)

    if dynamic_num_inference_steps:
        # Restore num_inference_steps to avoid confusion in the next generation if it is not dynamic
        scheduler.num_inference_steps = original_num_inference_steps

    # 9. Decode the final image.
    # Turn off fuser for typical SD
    gligen_enable_fuser(unet, False)
    images = decode(vae, latents)

    ret = [latents, images]

    if return_saved_cross_attn:
        ret.append(saved_attns)
    if return_box_vis:
        pil_images = [
            utils.draw_box(Image.fromarray(image), bboxes_item, phrases_item)
            for image, bboxes_item, phrases_item in zip(images, bboxes, phrases)
        ]
        ret.append(pil_images)
    if save_all_latents:
        latents_all = torch.stack(latents_all, dim=0)
        ret.append(latents_all)

    return tuple(ret)
