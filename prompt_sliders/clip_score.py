import argparse
import os
import re
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from tqdm import tqdm


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split(r'([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def compute_clip_score(image_path, prompt, processor, model, device):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        return outputs.logits_per_image[0][0].item()
    except Exception as e:
        print(f"[Error] {image_path}: {e}")
        return None


def extract_seed(folder_name):
    try:
        return int(folder_name.split("_")[-1])
    except (IndexError, ValueError):
        return None


def evaluate_folder(results_path, prompt, model_name, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model_folder = os.path.join(results_path, model_name)
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Model folder not found: {model_folder}")

    print(f"Evaluating folder: {model_folder}")
    seed_folders = sorted_nicely(os.listdir(model_folder))
    clip_data = []

    case_number = 0
    for folder_name in tqdm(seed_folders, desc=f"Processing {model_name}"):
        folder_path = os.path.join(model_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        seed = extract_seed(folder_name)
        if seed is None:
            print(f"[Skipping] Could not extract seed from folder: {folder_name}")
            continue

        row = {
            'case_number': case_number,
            'folder_name': folder_name,
            'seed': seed
        }

        for img_file in sorted_nicely(os.listdir(folder_path)):
            if not img_file.endswith('.png'):
                continue
            scale = img_file.replace('.png', '')  # e.g., '0', '0.5', etc.
            image_path = os.path.join(folder_path, img_file)
            score = compute_clip_score(image_path, prompt, processor, model, device)
            if score is not None:
                row[f'clip_{model_name}_{scale}'] = score

        clip_data.append(row)
        case_number += 1

    df = pd.DataFrame(clip_data)
    output_path = os.path.join(results_path, f'clip_scores_{model_name}.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved CLIP scores to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compute CLIP scores from edited image folders (case_number + seed)')
    parser.add_argument('--results_path', type=str, required=True, help='Path to results/image_editing folder.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to evaluate CLIP similarity against.')
    parser.add_argument('--model_name', type=str, required=True, help='Subfolder name under results/image_editing to process.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on.')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    evaluate_folder(
        results_path=args.results_path,
        prompt=args.prompt.strip(),
        model_name=args.model_name,
        device=device
    )


if __name__ == "__main__":
    main()
    