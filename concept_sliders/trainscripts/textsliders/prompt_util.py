import torch
import copy
import yaml

from typing import Literal, Optional, List, Union
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
