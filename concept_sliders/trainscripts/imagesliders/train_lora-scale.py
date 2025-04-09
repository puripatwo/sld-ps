import torch
import gc
import os, glob
import argparse
import ast
import wandb
import random
import numpy as np
from PIL import Image
from typing import Optional, List
from pathlib import Path
from tqdm import tqdm

import config_util
from config_util import RootConfig

import prompt_util
from prompt_util import PromptSettings, PromptEmbedsCache, PromptEmbedsPair

import model_util
import train_util
import debug_util
from lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV


def flush():
    torch.cuda.empty_cache()
    gc.collect()


def train(config: RootConfig, prompts: list[PromptSettings], device: int, folder_main: str, folders, scales):
    folders = np.array(folders)
    scales = np.array(scales)
    scales_unique = list(scales)

    # Create a prompt dictionary (1)
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    save_path = Path(config.save.path)
    if config.logging.verbose:
        print(metadata)

    # Set the network type (2)
    modules = DEFAULT_TARGET_REPLACE # ["Attention"]
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV
    
    # Initialize wandb (3)
    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)
    
    # Set the precision (4)
    weight_dtype = config_util.parse_precision(config.train.precision)
    save_weight_dtype = config_util.parse_precision(config.train.precision)

    # 1. Load the pre-trained models.
    tokenizer, text_encoder, unet, noise_scheduler, vae = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
    )

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval() # Freezes Text Encoder

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval() # Freezes UNet

    ### LoRA ###
    # 2. Set up LoRA.
    network = LoRANetwork(
        unet,
        rank=config.network.rank, # Controls LoRa's bottleneck size
        multiplier=1.0,
        alpha=config.network.alpha, # Scales the impact of LoRA
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    # 3. Optimizer setup.
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)
    criteria = torch.nn.MSELoss()

    # 4. Scheduler setup.
    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    
    print("Prompts: ")
    for settings in prompts:
        print(settings)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

    cache = PromptEmbedsCache()
    prompt_pairs: list[PromptEmbedsPair] = []

    # 5. Prompt encoding and caching.
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

                if isinstance(prompt, list):
                    if prompt == settings.positive:
                        key_setting = 'positive'
                    else:
                        key_setting = 'attributes'
                    if len(prompt) == 0:
                        cache[key_setting] = []
                    else:
                        if cache[key_setting] is None:
                            cache[key_setting] = train_util.encode_prompts(tokenizer, text_encoder, prompt)
                else:
                    if cache[prompt] == None:
                        cache[prompt] = train_util.encode_prompts(tokenizer, text_encoder, [prompt])

            prompt_pairs.append(PromptEmbedsPair(
                criteria,
                cache[settings.target],
                cache[settings.positive],
                cache[settings.unconditional],
                cache[settings.neutral],
                settings,
            ))

    del tokenizer
    del text_encoder
    flush()

    # 6. Start the training loop.
    pbar = tqdm(range(config.train.iterations))
    for i in pbar:
        with torch.no_grad():
            noise_scheduler.set_timesteps(
                config.train.max_denoising_steps, device=device
            )
            optimizer.zero_grad()

            # 6.1. Sampling a training example.
            prompt_pair: PromptEmbedsPair = prompt_pairs[
                torch.randint(0, len(prompt_pairs), (1,)).item()
            ]
            
            timesteps_to = torch.randint(1, config.train.max_denoising_steps - 1, (1,)).item()  # 1, 25, (1,)

            # # 6.2. Set up dynamic resolution.
            # height, width = (prompt_pair.resolution, prompt_pair.resolution)
            # if prompt_pair.dynamic_resolution:
            #     height, width = train_util.get_random_resolution_in_bucket(prompt_pair.resolution)
            
            # if config.logging.verbose:
            #     print("guidance_scale:", prompt_pair.guidance_scale)
            #     print("resolution:", prompt_pair.resolution)
            #     print("dynamic_resolution:", prompt_pair.dynamic_resolution)
            #     if prompt_pair.dynamic_resolution:
            #         print("bucketed resolution:", (height, width))
            #     print("batch_size:", prompt_pair.batch_size)
            
            # 6.3. Select an image pair from folders.
            scale_to_look = abs(random.choice(scales_unique))
            folder1 = folders[scales==-scale_to_look][0]  # folder1 = 'smallsize'
            folder2 = folders[scales==scale_to_look][0]  # folder2 = 'bigsize'

            ims = os.listdir(f'{folder_main}/{folder1}/')  # datasets/eyesize/smallsize/
            ims = [im_ for im_ in ims if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]

            # 6.4. Retrieve the input images.
            random_sampler = random.randint(0, len(ims)-1)
            img1 = Image.open(f'{folder_main}/{folder1}/{ims[random_sampler]}').resize((256,256))
            img2 = Image.open(f'{folder_main}/{folder2}/{ims[random_sampler]}').resize((256,256))

            seed = random.randint(0, 2 * 15)
            generator = torch.manual_seed(seed)

            # 6.5. Encode images into latents and add noise.
            denoised_latents_low, low_noise = train_util.get_noisy_image(
                img1,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_low = denoised_latents_low.to(device, dtype=weight_dtype)
            low_noise = low_noise.to(device, dtype=weight_dtype)

            generator = torch.manual_seed(seed)
            denoised_latents_high, high_noise = train_util.get_noisy_image(
                img2,
                vae,
                generator,
                unet,
                noise_scheduler,
                start_timesteps=0,
                total_timesteps=timesteps_to)
            denoised_latents_high = denoised_latents_high.to(device, dtype=weight_dtype)
            high_noise = high_noise.to(device, dtype=weight_dtype)

            # 6.6. Set to be in the same proportions as max_denoising_steps.
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            # Predicting noise of high latents
            high_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            # Predicting noise of low latents
            low_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("high_latents:", high_latents[0, 0, :5, :5])
                print("low_latents:", low_latents[0, 0, :5, :5])

        # 6.7. Train with positive scale.
        network.set_lora_slider(scale=scale_to_look)
        with network:
            # Predicting noise of target latents (high)
            target_latents_high = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_high,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("target_latents_high:", target_latents_high[0, 0, :5, :5])

        high_latents.requires_grad = False
        low_latents.requires_grad = False

        loss_high = criteria(target_latents_high, high_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_high.item()*1000:.4f}")
        loss_high.backward()

        # 6.8. Train with negative scale.
        network.set_lora_slider(scale=-scale_to_look)
        with network:
            # Predicting noise of target latents (low)
            target_latents_low = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents_low,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to("cpu", dtype=torch.float32)

            if config.logging.verbose:
                print("target_latents_low:", target_latents_low[0, 0, :5, :5])
            
        high_latents.requires_grad = False
        low_latents.requires_grad = False
        
        loss_low = criteria(target_latents_low, low_noise.cpu().to(torch.float32))
        pbar.set_description(f"Loss*1k: {loss_low.item()*1000:.4f}")
        loss_low.backward()

        optimizer.step()
        lr_scheduler.step()

        del (
            high_latents,
            low_latents,
            target_latents_low,
            target_latents_high,
        )
        flush()

        # 6.9. Saves model checkpoints periodically.
        if (i % config.save.per_steps == 0 and i != 0 and i != config.train.iterations - 1):
            print("Saving...")
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_{i}steps.safetensors",
                dtype=save_weight_dtype,
            )
    
    # 7. Final model saving after training ends.
    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
    )

    # 8. Clears memory and prints completion message.
    del (
        unet,
        noise_scheduler,
        loss,
        optimizer,
        network,
    )
    flush()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_util.load_config_from_yaml(config_file)
    if args.name is not None:
        config.save.name = args.name
    attributes = []
    if args.attributes is not None:
        attributes = args.attributes.split(',')
        attributes = [a.strip() for a in attributes]
    if args.prompts_file is not None:
        config.prompts_file = args.prompts_file
    if args.alpha is not None:
        config.network.alpha = args.alpha
    if args.rank is not None:
        config.network.rank = args.rank
    
    config.save.name += f'_alpha{config.network.alpha}'
    config.save.name += f'_rank{config.network.rank}'
    config.save.name += f'_{config.network.training_method}'  # eyeslider_alpha1.0_rank4_noxattn
    config.save.path += f'/{config.save.name}'  # ./models/eyeslider_alpha1.0_rank4_noxattn

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    print(prompts)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.device}")
    else:
        device = "cpu"

    folders = args.folders.split(',')
    folders = [f.strip() for f in folders]
    scales = args.scales.split(',')
    scales = [f.strip() for f in scales]
    scales = [int(s) for s in scales]

    print(folders, scales)
    if len(scales) != len(folders):
        raise Exception('the number of folders need to match the number of scales')
    
    if args.style_check is not None:
        check = args.style_check.split('-')
        for i in range(int(check[0]), int(check[1])):
            folder_main = args.folder_main + f'{i}'  # datasets/eyesize/{i}
            config.save.name = f'{os.path.basename(folder_main)}'  # {i}
            config.save.name += f'_alpha{args.alpha}'
            config.save.name += f'_rank{config.network.rank }'
            config.save.name += f'_{config.network.training_method}'  # {i}_alpha1.0_rank4_noxattn
            config.save.path = f'models/{config.save.name}'  # models/{i}_alpha1.0_rank4_noxattn
            train(config=config, prompts=prompts, device=device, folder_main=folder_main, folders=folders, scales=scales)
    else:
        train(config=config, prompts=prompts, device=device, folder_main=args.folder_main, folders=folders, scales=scales)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.") # config_file 'data/config.yaml'
    parser.add_argument("--prompts_file", required=False, help="Prompts file for training.", default=None)
    parser.add_argument("--alpha", type=float, required=False, default=None, help="LoRA weight.")  # --alpha 1.0
    parser.add_argument("--rank", type=int, required=False, help="Rank of LoRA.", default=None) # --rank 4
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to train on.") # --device 0
    parser.add_argument("--name", type=str, required=False, default=None, help="Name of the slider.") # --name 'eyesize_slider'
    parser.add_argument("--attributes", type=str, required=False, default=None, help="Attributes to disentangle (comma seperated string).") # --attributes 'male, female'
    parser.add_argument("--folder_main", type=str, required=True, help="The folder to check.")
    parser.add_argument("--folders", type=str, required=False, default='verylow, low, high, veryhigh', help="Folders with different attribute-scaled images.")
    parser.add_argument("--style_check", type=str, required=False, default=None, help="The style to check.")
    parser.add_argument("--scales", type=str, required=False, default='-2, -1, 1, 2', help="Scales for different attribute-scaled images.")
    args = parser.parse_args()

    main(args)
