# Apply prompt sliders for fine-grained control
# Reference: Sridhar et al., "Prompt Sliders: Fine-Grained Control of Text-to-Image Diffusion Models"
# https://arxiv.org/abs/2402.13946

import torch
import copy
import yaml

from typing import Literal, Optional, List, Union, Dict
from pathlib import Path
from pydantic import BaseModel, root_validator

ACTION_TYPES = Literal[
    "erase",
    "enhance",
]


class PromptSettings(BaseModel):
    target: str
    positive: str = None   # if None, target will be used
    unconditional: str = ""  # default is ""
    neutral: str = None  # if None, unconditional will be used

    action: ACTION_TYPES = "erase"  # default is "erase"
    guidance_scale: float = 1.0  # default is 1.0
    resolution: int = 512  # default is 512
    dynamic_resolution: bool = False  # default is False
    batch_size: int = 1  # default is 1
    dynamic_crops: bool = False  # default is False. only used when model is XL

    @root_validator(pre=True)
    def fill_prompts(cls, values):
        keys = values.keys()

        if "target" not in keys:
            raise ValueError("target must be specified")
        if "positive" not in keys:
            values["positive"] = values["target"]
        if "unconditional" not in keys:
            values["unconditional"] = ""
        if "neutral" not in keys:
            values["neutral"] = values["unconditional"]

        return values


def load_prompts_from_yaml(path, attributes = []):
    with open(path, "r") as f:
        prompts = yaml.safe_load(f)
    print(prompts)

    if len(prompts) == 0:
        raise ValueError("prompts file is empty")
    if len(attributes) != 0:
        new_prompts = []
        for i in range(len(prompts)):
            for att in attributes:
                copy_ = copy.deepcopy(prompts[i])
                copy_['target'] = att + ' ' + copy_['target']
                copy_['positive'] = att + ' ' + copy_['positive']
                copy_['neutral'] = att + ' ' + copy_['neutral']
                copy_['unconditional'] = att + ' ' + copy_['unconditional']
                new_prompts.append(copy_)
    else:
        new_prompts = copy.deepcopy(prompts)

    print(new_prompts)
    print(len(prompts), len(new_prompts))
    prompt_settings = [PromptSettings(**prompt) for prompt in new_prompts]

    return prompt_settings


class PromptEmbedsXL:
    text_embeds: torch.FloatTensor
    pooled_embeds: torch.FloatTensor

    def __init__(self, *args) -> None:
        self.text_embeds = args[0]
        self.pooled_embeds = args[1]


PROMPT_EMBEDDING = Union[torch.FloatTensor, PromptEmbedsXL]


class PromptEmbedsCache:  # 使いまわしたいので
    prompts: Dict[str, PROMPT_EMBEDDING] = {}

    def __setitem__(self, __name: str, __value: PROMPT_EMBEDDING) -> None:
        self.prompts[__name] = __value

    def __getitem__(self, __name: str) -> Optional[PROMPT_EMBEDDING]:
        if __name in self.prompts:
            return self.prompts[__name]
        else:
            return None
        

class PromptEmbedsPair:
    target: PROMPT_EMBEDDING  # not want to generate the concept
    positive: PROMPT_EMBEDDING  # generate the concept
    unconditional: PROMPT_EMBEDDING  # uncondition (default should be empty)
    neutral: PROMPT_EMBEDDING  # base condition (default should be empty)

    guidance_scale: float
    resolution: int
    dynamic_resolution: bool
    batch_size: int
    dynamic_crops: bool

    loss_fn: torch.nn.Module
    action: ACTION_TYPES

    def __init__(
        self,
        loss_fn: torch.nn.Module,
        target: PROMPT_EMBEDDING,
        positive: PROMPT_EMBEDDING,
        unconditional: PROMPT_EMBEDDING,
        neutral: PROMPT_EMBEDDING,
        settings: PromptSettings,
    ) -> None:
        self.loss_fn = loss_fn
        self.target = target
        self.positive = positive
        self.unconditional = unconditional
        self.neutral = neutral

        self.guidance_scale = settings.guidance_scale
        self.resolution = settings.resolution
        self.dynamic_resolution = settings.dynamic_resolution
        self.batch_size = settings.batch_size
        self.dynamic_crops = settings.dynamic_crops
        self.action = settings.action
        self.settings = settings

    def _erase(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        unconditional_latents: torch.FloatTensor,  # ""
        neutral_latents: torch.FloatTensor,  # ""
        scale: float = 1.0,
    ) -> torch.FloatTensor:
        
        """Target latents are going to not have the positive concept."""
        return self.loss_fn(target_latents, neutral_latents - scale * self.guidance_scale * (positive_latents - unconditional_latents))
    
    def _enhance(
        self,
        target_latents: torch.FloatTensor,  # "van gogh"
        positive_latents: torch.FloatTensor,  # "van gogh"
        unconditional_latents: torch.FloatTensor,  # ""
        neutral_latents: torch.FloatTensor,  # ""
        scale: float = 1.0,
    ):
        
        """Target latents are going to have the positive concept."""
        return self.loss_fn(target_latents, neutral_latents + scale * self.guidance_scale * (positive_latents - unconditional_latents))
    
    def loss(
        self,
        **kwargs,
    ):
        if self.action == "erase":
            return self._erase(**kwargs)
        elif self.action == "enhance":
            return self._enhance(**kwargs)
        else:
            raise ValueError("action must be erase or enhance")
        