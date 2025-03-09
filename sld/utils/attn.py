import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils


# image_generator.py
def shift_saved_attns_item(saved_attns_item, offset, guidance_attn_keys, horizontal_shift_only=False):
    """
    `horizontal_shift_only`: only shift horizontally. If you use `offset` from `compose_latents_with_alignment` with `horizontal_shift_only=True`, the `offset` already has y_offset = 0 and this option is not needed.
    """
    x_offset, y_offset = offset
    if horizontal_shift_only:
        y_offset = 0.
    
    new_saved_attns_item = {}
    for k in guidance_attn_keys:
        attn_map = saved_attns_item[k]
        
        attn_size = attn_map.shape[-2]
        attn_h = attn_w = int(math.sqrt(attn_size))
        # Example dimensions: [batch_size, num_heads, 8, 8, num_tokens]
        attn_map = attn_map.unflatten(2, (attn_h, attn_w))
        attn_map = utils.shift_tensor(
            attn_map, x_offset, y_offset, 
            offset_normalized=True, ignore_last_dim=True
        )
        attn_map = attn_map.flatten(2, 3)
        
        new_saved_attns_item[k] = attn_map
        
    return new_saved_attns_item


# image_generator.py
def shift_saved_attns(saved_attns, offset, guidance_attn_keys, **kwargs):
    # Iterate over timesteps
    shifted_saved_attns = [shift_saved_attns_item(saved_attns_item, offset, guidance_attn_keys, **kwargs) for saved_attns_item in saved_attns]
    
    return shifted_saved_attns