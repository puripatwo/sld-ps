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
        # print("checkpoint 1")
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu().float(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        conf_scores = outputs.iou_scores.cpu().numpy()[0, 0]
        del inputs, outputs
    
    # Uncomment if experiencing out-of-memory error:
    utils.free_memory()
    # print("checkpoint 2")
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


def sam_box_input(sam_model_dict, image, input_boxes, **kwargs):
    return sam(sam_model_dict, image, input_boxes=input_boxes, **kwargs)


def get_iou_with_resize(mask, masks, masks_shape):
    masks = np.array(
        [
            cv2.resize(
                mask.astype(np.uint8) * 255, masks_shape[::-1], cv2.INTER_LINEAR
            ).astype(bool)
            for mask in masks
        ]
    )
    return utils.iou(mask, masks)


def select_mask(
    masks,
    conf_scores,
    coarse_ious=None,
    rule="largest_over_conf",
    discourage_mask_below_confidence=0.85,
    discourage_mask_below_coarse_iou=0.2,
    verbose=False,
):
    """masks: numpy bool array"""
    mask_sizes = masks.sum(axis=(1, 2))

    # Another possible rule: iou with the attention mask
    if rule == "largest_over_conf":
        # Use the largest segmentation
        # Discourage selecting masks with conf too low or coarse iou is too low
        max_mask_size = np.max(mask_sizes)
        if coarse_ious is not None:
            scores = (
                mask_sizes
                - (conf_scores < discourage_mask_below_confidence) * max_mask_size
                - (coarse_ious < discourage_mask_below_coarse_iou) * max_mask_size
            )
        else:
            scores = (
                mask_sizes
                - (conf_scores < discourage_mask_below_confidence) * max_mask_size
            )
        if verbose:
            print(f"mask_sizes: {mask_sizes}, scores: {scores}")
    else:
        raise ValueError(f"Unknown rule: {rule}")

    mask_id = np.argmax(scores)
    mask = masks[mask_id]

    selection_conf = conf_scores[mask_id]

    if coarse_ious is not None:
        selection_coarse_iou = coarse_ious[mask_id]
    else:
        selection_coarse_iou = None

    if verbose:
        # print(f"Confidences: {conf_scores}")
        print(
            f"Selected a mask with confidence: {selection_conf}, coarse_iou: {selection_coarse_iou}"
        )

    if verbose >= 2:
        plt.figure(figsize=(10, 8))
        # plt.suptitle("After SAM")
        for ind in range(3):
            plt.subplot(1, 3, ind + 1)
            # This is obtained before resize.
            plt.title(
                f"Mask {ind}, score {scores[ind]}, conf {conf_scores[ind]:.2f}, iou {coarse_ious[ind] if coarse_ious is not None else None:.2f}"
            )
            plt.imshow(masks[ind])
        plt.tight_layout()
        plt.show()
        plt.close()

    return mask, selection_conf


def sam_refine_boxes(
    sam_input_images,
    boxes,
    model_dict,
    height,
    width,
    H,
    W,
    discourage_mask_below_confidence,
    discourage_mask_below_coarse_iou,
    verbose,
):
    # (w, h)
    input_boxes = [
        [utils.scale_proportion(box, H=height, W=width) for box in boxes_item]
        for boxes_item in boxes
    ]

    masks, conf_scores = sam_box_input(
        model_dict,
        image=sam_input_images,
        input_boxes=input_boxes,
        target_mask_shape=(H, W),
    )

    mask_selected_batched_list, conf_score_selected_batched_list = [], []

    for boxes_item, masks_item in zip(boxes, masks):
        mask_selected_list, conf_score_selected_list = [], []
        for box, three_masks in zip(boxes_item, masks_item):
            mask_binary = utils.proportion_to_mask(box, H, W, return_np=True)
            if verbose >= 2:
                # Also the box is the input for SAM
                plt.title("Binary mask from input box (for iou)")
                plt.imshow(mask_binary)
                plt.show()

            coarse_ious = get_iou_with_resize(
                mask_binary, three_masks, masks_shape=mask_binary.shape
            )

            mask_selected, conf_score_selected = select_mask(
                three_masks,
                conf_scores,
                coarse_ious=coarse_ious,
                rule="largest_over_conf",
                discourage_mask_below_confidence=discourage_mask_below_confidence,
                discourage_mask_below_coarse_iou=discourage_mask_below_coarse_iou,
                verbose=False,
            )

            mask_selected_list.append(mask_selected)
            conf_score_selected_list.append(conf_score_selected)
        mask_selected_batched_list.append(mask_selected_list)
        conf_score_selected_batched_list.append(conf_score_selected_list)

    return mask_selected_batched_list, conf_score_selected_batched_list


def sam_refine_box(sam_input_image, box, *args, **kwargs):
    # One image with one box

    sam_input_images, boxes = [sam_input_image], [[box]]
    mask_selected_batched_list, conf_score_selected_batched_list = sam_refine_boxes(
        sam_input_images, boxes, *args, **kwargs
    )

    return mask_selected_batched_list[0][0], conf_score_selected_batched_list[0][0]
