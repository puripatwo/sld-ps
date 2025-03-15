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
    # Create a prompt dictionary
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
    }
    save_path = Path(config.save.path)

    modules = DEFAULT_TARGET_REPLACE
    if config.network.type == "c3lier":
        modules += UNET_TARGET_REPLACE_MODULE_CONV

    if config.logging.verbose:
        print(metadata)

    # Initialize wandb
    if config.logging.use_wandb:
        wandb.init(project=f"LECO_{config.save.name}", config=metadata)

    # Set the precision
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
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    # 2. Set up LoRA.
    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)

    # 3. Optimizer and scheduler setup.
    optimizer_module = train_util.get_optimizer(config.train.optimizer)
    optimizer_kwargs = {}
    if config.train.optimizer_args is not None and len(config.train.optimizer_args) > 0:
        for arg in config.train.optimizer_args.split(" "):
            key, value = arg.split("=")
            value = ast.literal_eval(value)
            optimizer_kwargs[key] = value
    optimizer = optimizer_module(network.prepare_optimizer_params(), lr=config.train.lr, **optimizer_kwargs)

    lr_scheduler = train_util.get_lr_scheduler(
        config.train.lr_scheduler,
        optimizer,
        max_iterations=config.train.iterations,
        lr_min=config.train.lr / 100,
    )
    criteria = torch.nn.MSELoss()
    
    print("Prompts: ")
    for settings in prompts:
        print(settings)

    debug_util.check_requires_grad(network)
    debug_util.check_training_mode(network)

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
    config.save.name += f'_{config.network.training_method}'
    config.save.path += f'/{config.save.name}'

    prompts = prompt_util.load_prompts_from_yaml(config.prompts_file, attributes)
    print(prompts)
    device = torch.device(f"cuda:{args.device}")

    train(config, prompts, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.") # config_file 'data/config.yaml'
    parser.add_argument("--prompts_file", required=False, help="Prompts file for training.", default=None)
    parser.add_argument("--alpha", type=float, required=False, default=None, help="LoRA weight.")  # --alpha 1.0
    parser.add_argument("--rank", type=int, required=False, help="Rank of LoRA.", default=None) # --rank 4
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to train on.") # --device 0
    parser.add_argument("--name", type=str, required=False, default=None, help="Device to train on.") # --name 'eyesize_slider'
    parser.add_argument("--attributes", type=str, required=False, default=None, help="Attributes to disentangle (comma seperated string).") # --attributes 'male, female'
    args = parser.parse_args()

    main(args)
