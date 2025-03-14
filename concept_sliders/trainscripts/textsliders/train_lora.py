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
from prompt_util import PromptSettings

import model_util
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

    # 1. Load the models.
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

    # 2. Load in LoRA.
    network = LoRANetwork(
        unet,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        train_method=config.network.training_method,
    ).to(device, dtype=weight_dtype)


def main(args):
    config_file = args.config_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Config file for training.") # config_file 'data/config.yaml'
    parser.add_argument("--prompts_file", required=False, help="Prompts file for training.", default=None)
    parser.add_argument("--alpha", type=float, required=False, default=None, help="LoRA weight.")  # --alpha 1.0
    parser.add_argument("--rank", type=int, required=False, help="Rank of LoRA.", default=None) # --rank 4
    parser.add_argument("--device", type=int, required=False, default=0, help="Device to train on.") # --device 0
    parser.add_argument("--name", type=str, required=False, default=None, help="Device to train on.") # --name 'eyesize_slider'
    parser.add_argument("--attributes", type=str, required=False, default=None, help="attritbutes to disentangle (comma seperated string)") # --attributes 'male, female'
    args = parser.parse_args()

    main(args)
    