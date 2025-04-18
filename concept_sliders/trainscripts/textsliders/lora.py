import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

import os
import math
from typing import Optional, Literal, List, Type, Set

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

UNET_TARGET_REPLACE_MODULE_TRANSFORMER = [
    "Attention"
]
UNET_TARGET_REPLACE_MODULE_CONV = [
    "ResnetBlock2D",
    "Downsample2D",
    "Upsample2D",
    "DownBlock2D",
    "UpBlock2D",
]

DEFAULT_TARGET_REPLACE = UNET_TARGET_REPLACE_MODULE_TRANSFORMER

LORA_PREFIX_UNET = "lora_unet"


class LoRAModule(nn.Module):
    def __init__(
        self,
        lora_name,
        orig_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
    ):
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if "Linear" in orig_module.__class__.__name__:
            in_dim = orig_module.in_features
            out_dim = orig_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif "Conv" in orig_module.__class__.__name__:
            in_dim = orig_module.in_features
            out_dim = orig_module.out_features
            self.lora_dim = min(self.lora_dim, in_dim, out_dim)
            if self.lora_dim != lora_dim:
                print(f"{lora_name} dim (rank) is changed to: {self.lora_dim}.")
            kernel_size = orig_module.kernel_size
            stride = orig_module.stride
            padding = orig_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))

        nn.init.kaiming_uniform_(self.lora_down.weight, a=1)
        nn.init.zeros_(self.lora_up.weight)
        
        self.multiplier = multiplier
        self.orig_module = orig_module
    
    def apply_to(self):
        # Injecting the original module's forward pass with the new LoRA-enhanced one
        self.org_forward = self.orig_module.forward
        self.orig_module.forward = self.forward
        del self.orig_module

    def forward(self, x):
        # Performing the LoRA forward formula
        return (self.org_forward(x) + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale)


class LoRANetwork(nn.Module):
    def __init__(
        self,
        unet: UNet2DConditionModel,
        rank: int = 4,
        multiplier: float = 1.0,
        alpha: float = 1.0,
        train_method: TRAINING_METHODS = "full",
    ) -> None:
        super().__init__()
        self.lora_scale = 1
        self.lora_dim = rank
        self.multiplier = multiplier
        self.alpha = alpha
        
        self.module = LoRAModule

        self.unet_loras = self.create_modules(
            LORA_PREFIX_UNET,
            unet,
            DEFAULT_TARGET_REPLACE,
            self.lora_dim,
            self.multiplier,
            train_method=train_method,
        )
        print(f"Create LoRA for U-Net: {len(self.unet_loras)} modules.")

        lora_names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"Duplicated LoRA name: {lora.lora_name}. {lora_names}."
            lora_names.add(lora.lora_name)

        # Each LoRA module replaces the forward of its original layer, and is also registered so it shows up in .parameters() or .state_dict()
        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        del unet
        torch.cuda.empty_cache()

    def create_modules(
        self,
        prefix: str,
        root_module: nn.Module,
        target_replace_modules: List[str],
        rank: int,
        multiplier: float,
        train_method: TRAINING_METHODS,
    ) -> list:
        loras = []
        names = []
        for name, module in root_module.named_modules():
            if train_method == "noxattn" or train_method == "noxattn-hspace" or train_method == "noxattn-hspace-last":  # Cross Attention と Time Embed 以外学習
                if "attn2" in name or "time_embed" in name:
                    continue
            elif train_method == "innoxattn":
                if "attn2" in name:
                    continue
            elif train_method == "selfattn":
                if "attn1" not in name:
                    continue
            elif train_method == "xattn" or train_method == "xattn-strict":
                if "attn2" not in name:
                    continue
            elif train_method == "full":
                pass
            else:
                raise NotImplementedError(
                    f"train_method: {train_method} is not implemented."
                )
            
            if module.__class__.__name__ in target_replace_modules:
                for child_name, child_module in module.named_modules():
                    if child_module.__class__.__name__ in ["Linear", "Conv2d", "LoRACompatibleLinear", "LoRACompatibleConv"]:
                        if train_method == 'xattn-strict':
                            if 'out' in child_name:
                                continue
                        if train_method == 'noxattn-hspace':
                            if 'mid_block' not in name:
                                continue
                        if train_method == 'noxattn-hspace-last':
                            if 'mid_block' not in name or '.1' not in name or 'conv2' not in child_name:
                                continue
                        
                        # Instantiate a LoRAModule and store it
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")
                        lora = self.module(lora_name, child_module, multiplier, rank, self.alpha)

#                         print(f"{lora_name}")
#                         print(name, child_name)
#                         print(child_module.weight.shape)

                        if lora_name not in names:
                            loras.append(lora)
                            names.append(lora_name)

        return loras
    
    def prepare_optimizer_params(self):
        all_params = []

        # Gather only trainable LoRA parameters so that they can be passed to the optimizer
        if self.unet_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.unet_loras]
            param_data = {"params": params}
            all_params.append(param_data)

        return all_params
    
    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v
        
#         for key in list(state_dict.keys()):
#             if not key.startswith("lora"):
#                 # lora以外除外
#                 del state_dict[key]

        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)
    
    def set_lora_slider(self, scale):
        self.lora_scale = scale

    def __enter__(self):
        for lora in self.unet_loras:
            lora.multiplier = 1.0 * self.lora_scale

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras:
            lora.multiplier = 0
            