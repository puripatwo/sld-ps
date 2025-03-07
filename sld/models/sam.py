import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gc
import cv2

import utils
from utils import torch_device
from transformers import SamModel, SamProcessor


def load_sam():
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(torch_device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    sam_model_dict = {"sam_model": sam_model, "sam_processor": sam_processor}
    return sam_model_dict


# Not fully backward compatible with the previous implementation
# Reference: lmdv2/notebooks/gen_masked_latents_multi_object_ref_ca_loss_modular.ipynb
def sam(
    sam_model_dict,
    image,
    input_points=None,
    input_boxes=None,
    target_mask_shape=None,
    return_numpy=True,
):
    """target_mask_shape: (h, w)"""
    sam_model, sam_processor = (
        sam_model_dict["sam_model"],
        sam_model_dict["sam_processor"],
    )

    if not isinstance(input_boxes, torch.Tensor):
        if input_boxes and isinstance(input_boxes[0], tuple):
            # Convert tuple to list
            input_boxes = [list(input_box) for input_box in input_boxes]

        if input_boxes and input_boxes[0] and isinstance(input_boxes[0][0], tuple):
            # Convert tuple to list
            input_boxes = [
                [list(input_box) for input_box in input_boxes_item]
                for input_boxes_item in input_boxes
            ]
    
    with torch.no_grad():
        with torch.autocast(torch_device):
            inputs = sam_processor(
                image,
                input_points=input_points,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(torch_device)
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().float(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0, 0]
        del inputs, outputs
    
    # Uncomment if experiencing out-of-memory error:
    utils.free_memory()
    if return_numpy:
        masks = [
            F.interpolate(
                masks_item.type(torch.float), target_mask_shape, mode="bilinear"
            )
            .type(torch.bool)
            .numpy()
            for masks_item in masks
        ]
    else:
        masks = [
            F.interpolate(
                masks_item.type(torch.float), target_mask_shape, mode="bilinear"
            ).type(torch.bool)
            for masks_item in masks
        ]

    return masks, conf_scores