import numpy as np
import torch
import gc
import cv2
from PIL import ImageDraw

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"

# if torch.backends.mps.is_available():
#     torch_device = "mps"
# else:
#     torch_device = "cpu"


def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


# detector.py
def nms(
    bounding_boxes,
    confidence_score,
    labels,
    threshold,
    input_in_pixels=False,
    return_array=True,
):
    """
    This NMS processes boxes of all labels. It not only removes the box with the same label.

    Adapted from https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # coordinates of bounding boxes
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # Confidence scores of bounding boxes
    score = np.array(confidence_score)

    # Picked bounding boxes
    picked_boxes = []
    picked_score = []
    picked_labels = []

    # Compute areas of bounding boxes
    if input_in_pixels:
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    else:
        areas = (end_x - start_x) * (end_y - start_y)

    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]

        # Pick the bounding box with largest confidence score
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_labels.append(labels[index])

        # Compute ordinates of intersection-over-union(IOU)
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        # Compute areas of intersection-over-union
        if input_in_pixels:
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
        else:
            w = np.maximum(0.0, x2 - x1)
            h = np.maximum(0.0, y2 - y1)
        intersection = w * h

        # Compute the ratio between intersection and union
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(ratio < threshold)
        order = order[left]

    if return_array:
        picked_boxes, picked_score, picked_labels = (
            np.array(picked_boxes),
            np.array(picked_score),
            np.array(picked_labels),
        )

    return picked_boxes, picked_score, picked_labels


# detector.py
def post_process(box):
    new_box = []
    for item in box:
        item = min(1.0, max(0.0, item))
        new_box.append(round(item, 3))
    return new_box


# run_sld.py
def calculate_scale_ratio(region_a_param, region_b_param):
    _, _, a_width, a_height = region_a_param
    _, _, b_width, b_height = region_b_param
    scale_ratio_width = b_width / a_width
    scale_ratio_height = b_height / a_height
    return min(scale_ratio_width, scale_ratio_height)


# run_sld.py
def resize_image(image, region_a_param, region_b_param):
    """
    Resizes the image based on the scaling ratio between two regions and performs cropping or padding.
    """
    old_h, old_w, _ = image.shape
    scale_ratio = calculate_scale_ratio(region_a_param, region_b_param)

    new_size = (int(old_w * scale_ratio), int(old_h * scale_ratio))

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    new_h, new_w, _ = resized_image.shape
    region_a_param_real = [
        int(region_a_param[0] * new_h),
        int(region_a_param[1] * new_w),
        int(region_a_param[2] * new_h),
        int(region_a_param[3] * new_w),
    ]
    if scale_ratio >= 1:  # Cropping
        new_xmin = min(region_a_param_real[0], int(new_h - old_h))
        new_ymin = min(region_a_param_real[1], int(new_w - old_w))

        new_img = resized_image[
            new_ymin : new_ymin + old_w, new_xmin : new_xmin + old_h
        ]

        new_param = [
            (region_a_param_real[0] - new_xmin) / old_h,
            (region_a_param_real[1] - new_ymin) / old_w,
            region_a_param[2] * scale_ratio,
            region_a_param[3] * scale_ratio,
        ]
    else:  # Padding
        new_img = np.ones((old_h, old_w, 3), dtype=np.uint8) * 255
        new_img[:new_h, :new_w] = resized_image
        new_param = [region_a_param[i] * scale_ratio for i in range(4)]

    return new_img, new_param


# image_generator.py
def get_centered_box(
    box,
    horizontal_center_only=True,
    vertical_placement="centered",
    vertical_center=0.5,
    floor_padding=None,
):
    x_min, y_min, x_max, y_max = box
    w = x_max - x_min

    x_min_new = 0.5 - w / 2
    x_max_new = 0.5 + w / 2

    if horizontal_center_only:
        return [x_min_new, y_min, x_max_new, y_max]

    h = y_max - y_min

    if vertical_placement == "centered":
        assert (
            floor_padding is None
        ), "Set vertical_placement to floor_padding to use floor padding"

        y_min_new = vertical_center - h / 2
        y_max_new = vertical_center + h / 2
    elif vertical_placement == "floor_padding":
        # Ignores `vertical_center`

        y_max_new = 1 - floor_padding
        y_min_new = y_max_new - h
    else:
        raise ValueError(f"Unknown vertical placement: {vertical_placement}")

    return [x_min_new, y_min_new, x_max_new, y_max_new]


# sam.py
def scale_proportion(obj_box, H, W, use_legacy=False):
    if use_legacy:
        # Bias towards the top-left corner
        x_min, y_min, x_max, y_max = (
            int(obj_box[0] * W),
            int(obj_box[1] * H),
            int(obj_box[2] * W),
            int(obj_box[3] * H),
        )
    else:
        # Separately rounding box_w and box_h to allow shift invariant box sizes. Otherwise box sizes may change when both coordinates being rounded end with ".5".
        x_min, y_min = round(obj_box[0] * W), round(obj_box[1] * H)
        box_w, box_h = round((obj_box[2] - obj_box[0]) * W), round(
            (obj_box[3] - obj_box[1]) * H
        )
        x_max, y_max = x_min + box_w, y_min + box_h

        x_min, y_min = max(x_min, 0), max(y_min, 0)
        x_max, y_max = min(x_max, W), min(y_max, H)

    return x_min, y_min, x_max, y_max


# sam.py
def iou(mask, masks, eps=1e-6):
    # mask: [h, w], masks: [n, h, w]
    mask = mask[None].astype(bool)
    masks = masks.astype(bool)
    i = (mask & masks).sum(axis=(1, 2))
    u = (mask | masks).sum(axis=(1, 2))

    return i / (u + eps)


# sam.py
# NOTE: this changes the behavior of the function
def proportion_to_mask(obj_box, H, W, use_legacy=False, return_np=False):
    x_min, y_min, x_max, y_max = scale_proportion(obj_box, H, W, use_legacy)
    if return_np:
        mask = np.zeros((H, W))
    else:
        mask = torch.zeros(H, W).to(torch_device)
    mask[y_min:y_max, x_min:x_max] = 1.0

    return mask


# latents.py
def expand_overall_bboxes(overall_bboxes):
    """
    Expand overall bboxes from a 3d list to 2d list:
    Input: [[box 1 for phrase 1, box 2 for phrase 1], ...]
    Output: [box 1, box 2, ...]
    """
    return sum(overall_bboxes, start=[])


# latents.py
def binary_mask_to_center(mask, normalize=False):
    """
    This computes the mass center of the mask.
    normalize: the coords range from 0 to 1

    Reference: https://stackoverflow.com/a/66184125
    """
    h, w = mask.shape

    total = mask.sum()
    if isinstance(mask, torch.Tensor):
        x_coord = ((mask.sum(dim=0) @ torch.arange(w)) / total).item()
        y_coord = ((mask.sum(dim=1) @ torch.arange(h)) / total).item()
    else:
        x_coord = (mask.sum(axis=0) @ np.arange(w)) / total
        y_coord = (mask.sum(axis=1) @ np.arange(h)) / total

    if normalize:
        x_coord, y_coord = x_coord / w, y_coord / h
    return x_coord, y_coord


# attn.py
def shift_tensor(
    tensor,
    x_offset,
    y_offset,
    base_w=8,
    base_h=8,
    offset_normalized=False,
    ignore_last_dim=False,
):
    """base_w and base_h: make sure the shift is aligned in the latent and multiple levels of cross attention"""
    if ignore_last_dim:
        tensor_h, tensor_w = tensor.shape[-3:-1]
    else:
        tensor_h, tensor_w = tensor.shape[-2:]
    if offset_normalized:
        assert (
            tensor_h % base_h == 0 and tensor_w % base_w == 0
        ), f"{tensor_h, tensor_w} is not a multiple of {base_h, base_w}"
        scale_from_base_h, scale_from_base_w = tensor_h // base_h, tensor_w // base_w
        x_offset, y_offset = (
            round(x_offset * base_w) * scale_from_base_w,
            round(y_offset * base_h) * scale_from_base_h,
        )
    new_tensor = torch.zeros_like(tensor)

    overlap_w = tensor_w - abs(x_offset)
    overlap_h = tensor_h - abs(y_offset)

    if y_offset >= 0:
        y_src_start = 0
        y_dest_start = y_offset
    else:
        y_src_start = -y_offset
        y_dest_start = 0

    if x_offset >= 0:
        x_src_start = 0
        x_dest_start = x_offset
    else:
        x_src_start = -x_offset
        x_dest_start = 0

    if ignore_last_dim:
        # For cross attention maps, the third to last and the second to last are the 2D dimensions after unflatten.
        new_tensor[
            ...,
            y_dest_start : y_dest_start + overlap_h,
            x_dest_start : x_dest_start + overlap_w,
            :,
        ] = tensor[
            ...,
            y_src_start : y_src_start + overlap_h,
            x_src_start : x_src_start + overlap_w,
            :,
        ]
    else:
        new_tensor[
            ...,
            y_dest_start : y_dest_start + overlap_h,
            x_dest_start : x_dest_start + overlap_w,
        ] = tensor[
            ...,
            y_src_start : y_src_start + overlap_h,
            x_src_start : x_src_start + overlap_w,
        ]

    return new_tensor
