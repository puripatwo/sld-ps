# Apply self-correcting LLM-controlled diffusion (SLD)
# Reference: Wu et al., "Self-Correcting LLM-Controlled Diffusion Models"
# https://arxiv.org/abs/2309.16668

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

from .lmd import get_eval_info_from_prompt_lmd


class Evaluator:
    def __init__(self):
        self.processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
        self.model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda()


def nms(bounding_boxes, confidence_score, labels, threshold, input_in_pixels=False, return_array=True):
    """
    This NMS processes boxes of all labels. It not only removes the box with the same label.
    
    Adapted from https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Bounding boxes
    boxes = np.array(bounding_boxes)

    # Coordinates of bounding boxes
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
        picked_boxes, picked_score, picked_labels = np.array(picked_boxes), np.array(picked_score), np.array(picked_labels)

    return picked_boxes, picked_score, picked_labels


def class_aware_nms(bounding_boxes, confidence_score, labels, threshold, input_in_pixels=False):
    """
    This NMS processes boxes of each label individually.
    """
    # If no bounding boxes, return empty list
    if len(bounding_boxes) == 0:
        return np.array([]), np.array([]), np.array([])

    picked_boxes, picked_score, picked_labels = [], [], []

    labels_unique = np.unique(labels)
    for label in labels_unique:
        bounding_boxes_label = [bounding_box for i, bounding_box in enumerate(bounding_boxes) if labels[i] == label]
        confidence_score_label = [confidence_score_item for i, confidence_score_item in enumerate(confidence_score) if labels[i] == label]
        labels_label = [label] * len(bounding_boxes_label)
        picked_boxes_label, picked_score_label, picked_labels_label = nms(bounding_boxes_label, confidence_score_label, labels_label, threshold=threshold, input_in_pixels=input_in_pixels, return_array=False)
        picked_boxes += picked_boxes_label
        picked_score += picked_score_label
        picked_labels += picked_labels_label
    
    picked_boxes, picked_score, picked_labels = np.array(picked_boxes), np.array(picked_score), np.array(picked_labels)

    return picked_boxes, picked_score, picked_labels


def to_gen_box_format(box, width, height):
    # Input: xyxy, ranging from 0 to 1
    # Output: xywh, unnormalized (in pixels)
    x_min, y_min, x_max, y_max = box
    return [x_min * width, y_min * height, (x_max - x_min) * width, (y_max - y_min) * height]


def evaluate_with_boxes(boxes, eval_info, verbose=False):
    predicate = eval_info["predicate"]
    print("boxes:", boxes)
    return predicate(boxes, verbose)


@torch.no_grad()
def eval_prompt(prompt, path, evaluator, prim_score_threshold = 0.2, attr_score_threshold=0.45, nms_threshold = 0.5, use_class_aware_nms=False, verbose=False, use_cuda=True):
    texts, eval_info = get_eval_info_from_prompt_lmd(prompt)

    eval_type = eval_info["type"]
    if eval_type == "attribution":
        score_threshold = attr_score_threshold
    else:
        score_threshold = prim_score_threshold

    image = Image.open(path)
    inputs = evaluator.processor(text=texts, images=image, return_tensors="pt")
    if use_cuda:
        inputs = inputs.to("cuda")
    outputs = evaluator.model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    width, height = image.size
    target_sizes = torch.Tensor([[height, width]])
    if use_cuda:
        target_sizes = target_sizes.cuda()
    
    # Convert outputs (bounding boxes and class logits) to COCO API
    results = evaluator.processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)

    # Retrieve predictions for the first image for the corresponding text queries
    i = 0
    text = texts[i]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
    boxes = boxes.cpu()
    boxes = np.array([[x_min / width, y_min / height, x_max / width, y_max / height] for (x_min, y_min, x_max, y_max), score in zip(boxes, scores) if score >= score_threshold])
    labels = np.array([label.cpu().numpy() for label, score in zip(labels, scores) if score >= score_threshold])
    scores = np.array([score.cpu().numpy() for score in scores if score >= score_threshold])

    print("Post-NMS:")

    if use_class_aware_nms:
        boxes, scores, labels = class_aware_nms(boxes, scores, labels, nms_threshold)
    else:
        boxes, scores, labels = nms(boxes, scores, labels, nms_threshold)

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} ({label}) with confidence {round(score.item(), 3)} at location {box}")

    if verbose:
        print(f"prompt: {prompt}, texts: {texts}, boxes: {boxes}, labels: {labels}, eval_info: {eval_info}")

    det_boxes = [{"name": text[label], "bounding_box": to_gen_box_format(box, width, height), "score": score} for box, score, label in zip(boxes, scores, labels)]
    eval_success = evaluate_with_boxes(det_boxes, eval_info, verbose=verbose)

    return eval_type, eval_success
