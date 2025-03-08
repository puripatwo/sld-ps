import inflect
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
import numpy as np

img_dir = "imgs"
p = inflect.engine()


# run_sld.py
def prepare_annotations(boxes):
    if isinstance(boxes[0], dict):
        return [{"name": box["name"], "bbox": box["bounding_box"]} for box in boxes]
    else:
        return [
            {"name": box[0], "bbox": [int(x * 512) for x in box[1]]}
            for box in boxes
        ]


# run_sld.py
def draw_boxes(ax, anns):
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in anns:
        c = np.random.random((1, 3)) * 0.6 + 0.4
        [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
        poly = [
            [bbox_x, bbox_y],
            [bbox_x, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y + bbox_h],
            [bbox_x + bbox_w, bbox_y],
        ]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
        color.append(c)

        # print(ann)
        name = ann["name"] if "name" in ann else str(ann["category_id"])
        ax.text(
            bbox_x,
            bbox_y,
            name,
            style="italic",
            bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
        )

    p = PatchCollection(polygons, facecolor="none", edgecolors=color, linewidths=2)
    ax.add_collection(p)


# run_sld.py
def show_boxes(
    gen_boxes,
    additional_boxes=None,  # New parameter for the second set of boxes
    bg_prompt=None,
    neg_prompt=None,
    show=False,
    save=False,
    img=None,
    fname=None,
):
    if len(gen_boxes) == 0 and (additional_boxes is None or len(additional_boxes) == 0):
        return

    anns = prepare_annotations(gen_boxes)
    additional_anns = prepare_annotations(additional_boxes) if additional_boxes else []

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 5))

    # Plot for gen_boxes
    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title("Curr Layout", pad=20)
    draw_boxes(ax1, anns)

    # Plot for additional_boxes
    ax2.imshow(np.ones((512, 512, 3), dtype=np.uint8) * 255)
    ax2.axis("off")
    ax2.set_title("New Layout", pad=20)
    draw_boxes(ax2, additional_anns)

    # Add background prompt if present
    if bg_prompt is not None:
        for ax in [ax1, ax2]:
            ax.text(
                0,
                0,
                bg_prompt + f" (Neg: {neg_prompt})" if neg_prompt else bg_prompt,
                style="italic",
                bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
                fontsize=8,
            )

    if show:
        plt.show()
    else:
        print("Saved boxes visualizations to", f"{fname}")
        plt.savefig(fname)
    plt.clf()


# image_generator.py
def convert_box(box, height, width):
    # box: x, y, w, h (in 512 format) -> x_min, y_min, x_max, y_max
    x_min, y_min = box[0] / width, box[1] / height
    w_box, h_box = box[2] / width, box[3] / height

    x_max, y_max = x_min + w_box, y_min + h_box

    return x_min, y_min, x_max, y_max


# image_generator.py
def convert_spec(spec, height, width, include_counts=True, verbose=False):
    # Infer from spec
    prompt, gen_boxes, bg_prompt = spec["prompt"], spec["gen_boxes"], spec["bg_prompt"]

    # This ensures the same objects appear together because flattened `overall_phrases_bboxes` should EXACTLY correspond to `so_prompt_phrase_box_list`.
    gen_boxes = sorted(gen_boxes, key=lambda gen_box: gen_box[0])

    gen_boxes = [
        (name, convert_box(box, height=height, width=width)) for name, box in gen_boxes
    ]

    # NOTE: so phrase should include all the words associated to the object (otherwise "an orange dog" may be recognized as "an orange" by the model generating the background).
    # so word should have one token that includes the word to transfer cross attention (the object name).
    # Currently using the last word of the object name as word.
    if bg_prompt:
        so_prompt_phrase_word_box_list = [
            (f"{bg_prompt} with {name}", name, name.split(" ")[-1], box)
            for name, box in gen_boxes
        ]
    else:
        so_prompt_phrase_word_box_list = [
            (f"{name}", name, name.split(" ")[-1], box) for name, box in gen_boxes
        ]

    objects = [gen_box[0] for gen_box in gen_boxes]

    objects_unique, objects_count = np.unique(objects, return_counts=True)

    num_total_matched_boxes = 0
    overall_phrases_words_bboxes = []
    for ind, object_name in enumerate(objects_unique):
        bboxes = [box for name, box in gen_boxes if name == object_name]

        if objects_count[ind] > 1:
            phrase = p.plural_noun(object_name.replace("an ", "").replace("a ", ""))
            if include_counts:
                phrase = p.number_to_words(objects_count[ind]) + " " + phrase
        else:
            phrase = object_name
        # Currently using the last word of the phrase as word.
        word = phrase.split(" ")[-1]

        num_total_matched_boxes += len(bboxes)
        overall_phrases_words_bboxes.append((phrase, word, bboxes))

    assert num_total_matched_boxes == len(
        gen_boxes
    ), f"{num_total_matched_boxes} != {len(gen_boxes)}"

    objects_str = ", ".join([phrase for phrase, _, _ in overall_phrases_words_bboxes])
    if objects_str:
        if bg_prompt:
            overall_prompt = f"{bg_prompt} with {objects_str}"
        else:
            overall_prompt = objects_str
    else:
        overall_prompt = bg_prompt

    if verbose:
        print("so_prompt_phrase_word_box_list:", so_prompt_phrase_word_box_list)
        print("overall_prompt:", overall_prompt)
        print("overall_phrases_words_bboxes:", overall_phrases_words_bboxes)

    return so_prompt_phrase_word_box_list, overall_prompt, overall_phrases_words_bboxes