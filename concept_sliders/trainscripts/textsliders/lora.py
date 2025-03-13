import torch
import torch.nn as nn
from diffusers import UNet2DConditionalModel
from safetensors.torch import save_file

import os
import math
from typing import Optional, Literal, List

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    "xattn-strict", # q and k values
    "noxattn-hspace",
    "noxattn-hspace-last",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]
