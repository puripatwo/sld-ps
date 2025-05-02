# Apply prompt sliders for fine-grained control
# Reference: Sridhar et al., "Prompt Sliders: Fine-Grained Control of Text-to-Image Diffusion Models"
# https://arxiv.org/abs/2402.13946

import argparse
import logging
import math
import os
import random
import shutil
import PIL
from pathlib import Path
from typing import List, Optional
from packaging import version
from tqdm.auto import tqdm

import numpy as np
import safetensors
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset
from torchvision import transforms

import transformers
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available

import train_util
import prompt_util
from prompt_util import PromptEmbedsCache, PromptEmbedsPair, PromptSettings, PromptEmbedsXL

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }

check_min_version("0.27.0.dev0")

logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path, safe_serialization=True):
    logger.info("Saving embeddings...")

    learned_embeds = (accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1])
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}

    if safe_serialization:
        safetensors.torch.save_file(learned_embeds_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(learned_embeds_dict, save_path)


def log_validation(text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae, args, accelerator, weight_dtype, epoch, is_final_validation=False):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )

    # Create a pipeline for inference
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder_1),
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer_1,
        tokenizer_2=tokenizer_2,
        unet=unet,
        vae=vae,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # Run inference
    images = []
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
        images.append(image)

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(tracker_key, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({tracker_key: [wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)]})
        
    del pipeline
    torch.cuda.empty_cache()
    return images


def save_model_card(repo_id: str, images: list = None, base_model: str = None, repo_folder: str = None):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"
    
    model_description = f"""
    Textual inversion text2image fine-tuning - {repo_id}
    These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
    {img_str}
    """
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )
    tags = ["stable-diffusion-xl", "stable-diffusion-xl-diffusers", "text-to-image", "diffusers", "textual_inversion"]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_steps", type=int, default=500, help="Save learned_embeds.bin every X updates steps.")
    parser.add_argument("--save_as_full_pipeline", action="store_true", help="Save the complete stable diffusion pipeline.")
    parser.add_argument("--num_vectors", type=int, default=1, help="How many textual inversion vectors shall be used to learn the concept.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default=None, required=True, help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False, help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None, help="Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g., fp16.")
    # parser.add_argument("--tokenizer_name", type=str, default=None, help="Pretrained tokenizer name or path if not the same as model_name.") ###
    # parser.add_argument("--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data.")
    parser.add_argument("--placeholder_token", type=str, default=None, required=True, help="A token to use as a placeholder for the concept.")
    parser.add_argument("--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word.")
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'.") ###
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument("--output_dir", type=str, default="text-inversion-model", help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.") ###
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution.")
    parser.add_argument("--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_train_epochs", type=int, default=100) ###
    parser.add_argument("--max_train_steps", type=int, default=5000, help="Total number of training steps to perform.  If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")
    parser.add_argument("--lr_scheduler", type=str, default="constant", help='The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].')
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1, help="Number of hard resets of the lr in cosine_with_restarts scheduler.")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.")
    parser.add_argument("--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes.")
    parser.add_argument("--prompts_file", type=str, default=None, help="prompt file.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The name of the repository to keep in sync with the local `output_dir`.")
    parser.add_argument("--logging_dir", type=str, default="logs", help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"], help="Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices.")
    parser.add_argument("--report_to", type=str, default="tensorboard", help='The integration to report the results and logs to. Supported platforms are `"tensorboard"` (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.')
    parser.add_argument("--validation_prompt", type=str, default=None, help="A prompt that is used during validation to verify that the model is learning.")
    parser.add_argument("--num_validation_images", type=int, default=4, help="Number of images that should be generated during validation with `validation_prompt`.")
    parser.add_argument("--validation_steps", type=int, default=100, help="Run validation every X steps. Validation consists of running the prompt `args.validation_prompt` multiple times: `args.num_validation_images` and logging the images.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank.") ###
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Max number of checkpoints to store.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or \"latest\" to automatically select the last available checkpoint.")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--no_safe_serialization", action="store_true", help="If specified save the checkpoint not in `safetensors` format, but in original PyTorch format instead.") ###
    args = parser.parse_args()
    
    return args


def train(args):
    # Check if LOCAL_RANK is set (1)
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    
    # Check if WandB is available (2)
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            "Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    # If passed along, set the training seed now (3)
    if args.seed is not None:
        set_seed(args.seed)
    
    # 0. Set up Accelerator.
    # 0.1. Initialize the accelerator.
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # 0.2. Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    if accelerator.is_main_process:
        logging.info(accelerator.state)

    # 0.3. Only the main process should show informational or warning logs.
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # 0.4. Handle the repository creation; use is_main_process to ensure that only the main process does this.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        
        # 0.5. Create the repo if it doesn't exist.
        if args.push_to_hub:
            repo_id = create_repo(repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token).repo_id
        
    # 1. Load the pre-trained models.
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_1 = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    text_encoder_2 = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant)

    ### TEXTUAL INVERSION ###
    # 2. Set up Textual Inversion.
    # 2.1 Add the placeholder token in tokenizer.
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}.")
    
    # 2.2. Add additional tokens for each vector.
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    # 2.3. Add the placeholder token to the tokenizer.
    num_added_tokens = tokenizer_1.add_tokens(placeholder_tokens)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # 2.4. Convert the initializer and placeholder tokens to IDs.
    token_ids = tokenizer_1.encode(args.initializer_token, add_special_tokens=False)
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")
    initializer_token_id = token_ids[0]

    placeholder_token_ids = tokenizer_1.convert_tokens_to_ids(placeholder_tokens)
    placeholder_token = (" ".join(tokenizer_1.convert_ids_to_tokens(placeholder_token_ids)))

    # 2.5. Resize the token embeddings of the text encoder.
    text_encoder_1.resize_token_embeddings(len(tokenizer_1))

    # 2.6. Initialise the newly added placeholder token with the embeddings of the initializer token.
    token_embeds = text_encoder_1.get_input_embeddings().weight.data
    with torch.no_grad():
        for placeholder_token_id in placeholder_token_ids:
            token_embeds[placeholder_token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # Freeze all parameters except for the token embeddings in text encoder
    text_encoder_1.text_model.encoder.requires_grad_(False)
    text_encoder_1.text_model.final_layer_norm.requires_grad_(False)
    text_encoder_1.text_model.embeddings.position_embedding.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    if args.gradient_checkpointing:
        text_encoder_1.gradient_checkpointing_enable()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers
            
            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # 3. Optimizer setup.
    optimizer = optimizer_class(
        text_encoder_1.get_input_embeddings().parameters(),  # only optimize the embeddings
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )
    criteria = torch.nn.MSELoss()

    tokenizers = [tokenizer_1, tokenizer_2]
    text_encoders = [text_encoder_1, text_encoder_2]

    attributes = []
    prompts = prompt_util.load_prompts_from_yaml(args.prompts_file, attributes)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    # 4. Prompt encoding and caching.
    with torch.no_grad():
        for settings in prompts:
            print(settings)

            for prompt in [
                settings.target,
                settings.positive,
                settings.neutral,
                settings.unconditional,
            ]:
                print(prompt)
                
                if cache[prompt] == None:
                    text_embeds, pooled_embeds = train_util.encode_prompts_xl(tokenizers, text_encoders, [prompt], num_images_per_prompt=1)
                    cache[prompt] = PromptEmbedsXL(
                        text_embeds,
                        pooled_embeds
                    )

            prompt_pairs.append(PromptEmbedsPair(
                criteria,
                cache[settings.target],
                cache[settings.positive],
                cache[settings.unconditional],
                cache[settings.neutral],
                settings,
            ))

    # 5. Scheduler setup.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(args.max_train_steps / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
    )

    # Set the text encoder to train mode (1)
    text_encoder_1.train()
    text_encoder_1, optimizer, lr_scheduler = accelerator.prepare(text_encoder_1, optimizer, lr_scheduler)

    # For mixed precision training, we cast all non-trainable weights to half-precision as these weights are only used for inference, keeping weights in full precision is not required (2)
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype (3)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed (4)
    num_update_steps_per_epoch = math.ceil(1 / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration (5)
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Calculate the total batch size (6)
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** TRAINING *****")
    logger.info(f"  Total Optimizer Steps = {args.max_train_steps}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Train Batch Size (per device) = {args.train_batch_size}")
    logger.info(f"  Total Train Batch Size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation Steps = {args.gradient_accumulation_steps}")

    # 6. Potentially load in the weights and states from a previous save.
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Keep the original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight.data.clone()

    # 7. Start the training loop.
    device = torch.device("cuda:0")
    pbar = tqdm(range(0, args.max_train_steps), initial=initial_global_step, desc="Steps", disable=not accelerator.is_local_main_process)
    for epoch in range(first_epoch, args.num_train_epochs):
        text_encoder_1.train()

        with torch.no_grad():
            noise_scheduler.set_timesteps(
                50, device=device
            )
            optimizer.zero_grad()

            # 7.1. Sampling a training example.
            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]

            timesteps_to = torch.randint(1, 50, (1,)).item()

            # 7.2. Set up dynamic resolution.
            height, width = (prompt_pair.resolution, prompt_pair.resolution)
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(prompt_pair.resolution)

            # if config.logging.verbose:
            #     print("guidance_scale:", prompt_pair.guidance_scale)
            #     print("resolution:", prompt_pair.resolution)
            #     print("dynamic_resolution:", prompt_pair.dynamic_resolution)
            #     if prompt_pair.dynamic_resolution:
            #         print("bucketed resolution:", (height, width))
            #     print("batch_size:", prompt_pair.batch_size)

            # 7.3. Generating latents.
            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(device, dtype=weight_dtype)

            # 7.4. Prepare a time embedding input vector for the UNet.
            add_time_ids = train_util.get_add_time_ids(
                height,
                width,
                dynamic_crops=prompt_pair.dynamic_crops,
                dtype=weight_dtype,
            ).to(device, dtype=weight_dtype)

            # 7.5. Modify the text embedding.
            new_target = prompt_pair.settings.target + f', {placeholder_token}'
            sc = float(random.choice([idx for idx in range(11)]))
            ti_prompt_1, ti_pool_embs_1 = train_util.encode_prompts_xl_slider(tokenizers, text_encoders, [new_target], sc=sc)

            # 7.6. Denoising process.
            denoised_latents = train_util.diffusion_xl(
                unet,
                noise_scheduler,
                latents, 
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds.to(device),
                    ti_prompt_1,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds.to(device),
                    ti_pool_embs_1,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                start_timesteps=0,
                total_timesteps=timesteps_to,
                guidance_scale=3,
            )

            # 7.7. Set to be in the same proportions as max_denoising_steps.
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / 50)
            ]

            # Predicting noise of positive latents
            positive_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.positive.text_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.positive.pooled_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            # Predicting noise of neutral latents
            neutral_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.neutral.text_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.neutral.pooled_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            # Predicting noise of unconditional latents
            unconditional_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.unconditional.text_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.unconditional.pooled_embeds,
                    prompt_pair.batch_size,
                ).to(device),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
        
        with accelerator.accumulate(text_encoder_1):
            ti_prompt_2, ti_pool_embs_2 = train_util.encode_prompts_xl_slider(tokenizers, text_encoders, [new_target], sc=sc)

            # Predicting noise of target latents
            target_latents = train_util.predict_noise_xl(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.text_embeds.to(device),
                    ti_prompt_2,
                    prompt_pair.batch_size,
                ),
                add_text_embeddings=train_util.concat_embeddings(
                    prompt_pair.unconditional.pooled_embeds.to(device),
                    ti_pool_embs_2,
                    prompt_pair.batch_size,
                ),
                add_time_ids=train_util.concat_embeddings(
                    add_time_ids, add_time_ids, prompt_pair.batch_size
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)
        
            positive_latents.requires_grad = False
            neutral_latents.requires_grad = False
            unconditional_latents.requires_grad = False

            # 7.8. Calculating the loss.
            loss = prompt_pair.loss(
                target_latents=target_latents,
                positive_latents=positive_latents,
                neutral_latents=neutral_latents,
                unconditional_latents=unconditional_latents,
                scale=sc,
            )

            # 7.9. Performing backpropagation to update the text embeddings.
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # 7.10. Make sure we don't update any embedding weights besides the newly added token.
            index_no_updates = torch.ones((len(tokenizer_1),), dtype=torch.bool)
            index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False
            with torch.no_grad():
                accelerator.unwrap_model(text_encoder_1).get_input_embeddings().weight[index_no_updates] = orig_embeds_params[index_no_updates]

            # 7.11. Check if the accelerator has performed an optimization step behind the scenes.
            if accelerator.sync_gradients:
                images = []
                pbar.update(1)
                global_step += 1

                # Saving the learned embeddings
                if global_step % args.save_steps == 0:
                    weight_name = (f"learned_embeds-steps-{global_step}.bin" if args.no_safe_serialization else f"learned_embeds-steps-{global_step}.safetensors")
                    save_path = os.path.join(args.output_dir, weight_name)
                    save_progress(
                        text_encoder_1,
                        placeholder_token_ids,
                        accelerator,
                        args,
                        save_path,
                        safe_serialization=not args.no_safe_serialization,
                    )

                # Saving the checkpoints
                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # Before we save the new checkpoint, we need to have at most `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints")
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)
                        
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    # Generate and log validation images
                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae, args, accelerator, weight_dtype, epoch)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            pbar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # 8. Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.validation_prompt:
            images = log_validation(text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2, unet, vae, args, accelerator, weight_dtype, epoch)
        if args.push_to_hub and not args.save_as_full_pipeline:
            logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
            save_full_model = True
        else:
            save_full_model = args.save_as_full_pipeline
        if save_full_model:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                text_encoder=accelerator.unwrap_model(text_encoder_1),
                text_encoder_2=text_encoder_2,
                vae=vae,
                unet=unet,
                tokenizer=tokenizer_1,
                tokenizer_2=tokenizer_2,
            )
            pipeline.save_pretrained(args.output_dir)
        
        # 9. Save the newly trained embeddings.
        weight_name = "learned_embeds.bin" if args.no_safe_serialization else "learned_embeds.safetensors"
        save_path = os.path.join(args.output_dir, weight_name)
        save_progress(
            text_encoder_1,
            placeholder_token_ids,
            accelerator,
            args,
            save_path,
            safe_serialization=not args.no_safe_serialization,
        )

        # 10. Push to the Hub if necessary.
        if args.push_to_hub:
            save_model_card(
                repo_id,
                images=images,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
    
    accelerator.end_training()
    print("Done.")


if __name__ == "__main__":
    args = parse_args()

    train(args)
