import torch
import gc
import argparse
import ast
import wandb
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


def train(config: RootConfig, prompts: list[PromptSettings], device: int):
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
    tokenizer, text_encoder, unet, noise_scheduler = model_util.load_models(
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
            
            timesteps_to = torch.randint(1, config.train.max_denoising_steps, (1,)).item()

            # 6.2. Set up dynamic resolution.
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

            # 6.3. Generating latents.
            latents = train_util.get_initial_latents(
                noise_scheduler, prompt_pair.batch_size, height, width, 1
            ).to(device, dtype=weight_dtype)

            # 6.4. Denoising process.
            with network:
                denoised_latents = train_util.diffusion(
                    unet,
                    noise_scheduler,
                    latents,  # 単純なノイズのlatentsを渡す
                    train_util.concat_embeddings(
                        prompt_pair.unconditional,
                        prompt_pair.target,
                        prompt_pair.batch_size,
                    ),
                    start_timesteps=0,
                    total_timesteps=timesteps_to,
                    guidance_scale=3,
                )

            # 6.5. Set to be in the same proportions as max_denoising_steps.
            noise_scheduler.set_timesteps(1000)
            current_timestep = noise_scheduler.timesteps[
                int(timesteps_to * 1000 / config.train.max_denoising_steps)
            ]

            # Predicting noise of positive latents
            positive_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.positive,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            # Predicting noise of neutral latents
            neutral_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.neutral,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            # Predicting noise of unconditional latents
            unconditional_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.unconditional,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            if config.logging.verbose:
                print("positive_latents:", positive_latents[0, 0, :5, :5])
                print("neutral_latents:", neutral_latents[0, 0, :5, :5])
                print("unconditional_latents:", unconditional_latents[0, 0, :5, :5])
        
        # Predicting noise of target latents
        with network:
            target_latents = train_util.predict_noise(
                unet,
                noise_scheduler,
                current_timestep,
                denoised_latents,
                train_util.concat_embeddings(
                    prompt_pair.unconditional,
                    prompt_pair.target,
                    prompt_pair.batch_size,
                ),
                guidance_scale=1,
            ).to(device, dtype=weight_dtype)

            if config.logging.verbose:
                print("target_latents:", target_latents[0, 0, :5, :5])

        positive_latents.requires_grad = False
        neutral_latents.requires_grad = False
        unconditional_latents.requires_grad = False

        # 6.6. Calculating the loss.
        loss = prompt_pair.loss(
            target_latents=target_latents,
            positive_latents=positive_latents,
            neutral_latents=neutral_latents,
            unconditional_latents=unconditional_latents,
        )

        # 6.7. Logs loss and learning rate to wandb for monitoring.
        pbar.set_description(f"Loss*1k: {loss.item()*1000:.4f}")
        if config.logging.use_wandb:
            wandb.log({"loss": loss, "iteration": i, "lr": lr_scheduler.get_last_lr()[0]})

        # 6.8. Performing backpropagation to update LoRA parameters.
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del (
            positive_latents,
            neutral_latents,
            unconditional_latents,
            target_latents,
            latents,
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

    # train(config, prompts, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.") # config_file 'data/config.yaml'
    parser.add_argument("--prompts_file", required=False, help="Prompts file for training.", default=None)
    parser.add_argument("--alpha", type=float, required=False, default=None, help="LoRA weight.")  # --alpha 1.0
    parser.add_argument("--rank", type=int, required=False, help="Rank of LoRA.", default=None) # --rank 4
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to train on.") # --device 0
    parser.add_argument("--name", type=str, required=False, default=None, help="Name of the slider.") # --name 'eyesize_slider'
    parser.add_argument("--attributes", type=str, required=False, default=None, help="Attributes to disentangle (comma seperated string).") # --attributes 'male, female'
    args = parser.parse_args()

    main(args)
