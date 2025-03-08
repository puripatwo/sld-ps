import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gc
import cv2
from transformers import SamModel, SamProcessor

import utils
from utils import torch_device


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
    
    print("Running SAM...")
    with torch.no_grad():
        with torch.autocast(torch_device):
            inputs = sam_processor(
                image,
                input_points=input_points,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(torch_device)
            outputs = sam_model(**inputs)
        print("checkpoint 1")
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().float(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0, 0]
        del inputs, outputs
    
    # Uncomment if experiencing out-of-memory error:
    utils.free_memory()
    print("checkpoint 2")
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


def run_sam(bbox, image_source, models):
    H, W, _ = image_source.shape
    box_xyxy = torch.Tensor(
        [
            bbox[0],
            bbox[1],
            bbox[2] + bbox[0],
            bbox[3] + bbox[1]
        ]
    ) * torch.Tensor([W, H, W, H])
    box_xyxy = box_xyxy.unsqueeze(0).unsqueeze(0)
    masks, _ = sam(
        models.model_dict,
        image_source,
        input_boxes=box_xyxy,
        target_mask_shape=(H, W),
    )
    masks = masks[0][0].transpose(1, 2, 0).astype(bool)
    return masks


def run_sam_postprocess(remove_mask, H, W, config):
    remove_mask = np.mean(remove_mask, axis=2)
    remove_mask[remove_mask > 0.05] = 1.0
    k_size = int(config.get("SLD", "SAM_refine_dilate"))
    kernel = np.ones((k_size, k_size), np.uint8)
    dilated_mask = cv2.dilate(
        (remove_mask * 255).astype(np.uint8), kernel, iterations=1
    )
    # Resize the mask from the image size to the latent size
    remove_region = cv2.resize(
        dilated_mask.astype(np.int64),
        dsize=(W // 8, H // 8),
        interpolation=cv2.INTER_NEAREST,
    )
    return remove_region
