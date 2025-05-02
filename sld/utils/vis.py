# Apply self-correcting LLM-controlled diffusion (SLD)
# Reference: Wu et al., "Self-Correcting LLM-Controlled Diffusion Models"
# https://arxiv.org/abs/2309.16668

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw

import utils
from . import parse

save_ind = 0


def reset_save_ind():
    global save_ind
    save_ind = 0


# image_generator.py
def display(image, save_prefix="", ind=None, save_ind_in_filename=True):
    """
    save_ind_in_filename: This adds a global index to the filename so that two calls to this function will not save to the same file and overwrite the previous image.
    """
    global save_ind
    if save_prefix != "":
        save_prefix = save_prefix + "_"
    if save_ind_in_filename:
        ind = f"{ind}_" if ind is not None else ""
        path = f"{parse.img_dir}/{save_prefix}{ind}{save_ind}.png"
    else:
        ind = f"{ind}" if ind is not None else ""
        path = f"{parse.img_dir}/{save_prefix}{ind}.png"

    print(f"Saved to {path}")

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image.save(path)
    save_ind = save_ind + 1
