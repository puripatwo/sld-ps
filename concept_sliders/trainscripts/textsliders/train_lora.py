import torch
import gc

import config_util
from config_util import RootConfig

import prompt_util
from prompt_util import PromptSettings

def flush():
    torch.cuda.empty_cache()
    gc.collect()

def train(config: RootConfig, prompts: list[PromptSettings], device: int):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
    }
    