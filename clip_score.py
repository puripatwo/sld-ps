import argparse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


def compute_clip_score(image_path, prompt, device):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        score = outputs.logits_per_image[0][0].item()
        print(f"CLIP score for '{image_path}' with prompt '{prompt}': {score:.4f}")
        return score
    except Exception as e:
        print(f"[Error] Could not process {image_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Compute CLIP score for a single image.')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the image.')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt to evaluate CLIP similarity against.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to run on.')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    compute_clip_score(args.image_path, args.prompt.strip(), device)


if __name__ == "__main__":
    main()
    