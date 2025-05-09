# Apply self-correcting LLM-controlled diffusion (SLD)
# Reference: Wu et al., "Self-Correcting LLM-Controlled Diffusion Models"
# https://arxiv.org/abs/2309.16668
# Template for self correction tasks --> adjust the bounding boxes
spot_difference_template = """# Your Role: Expert Bounding Box Adjuster

## Objective: Manipulate bounding boxes in square images according to the user prompt while maintaining visual accuracy.

## Bounding Box Specifications and Manipulations
1. Image Coordinates: Define square images with top-left at [0, 0] and bottom-right at [1, 1].
2. Box Format: [Top-left x, Top-left y, Width, Height]
3. Operations: Include addition, deletion, repositioning, and attribute modification.

## Key Guidelines
1. Alignment: Follow the user's prompt, keeping the specified object count and attributes. Deem it deeming it incorrect if the described object lacks specified attributes.
2. Boundary Adherence: Keep bounding box coordinates within [0, 1].
3. Minimal Modifications: Change bounding boxes only if they don't match the user's prompt (i.e., don't modify matched objects).
4. Overlap Reduction: Minimize intersections in new boxes and remove the smallest, least overlapping objects.

## Process Steps
1. Interpret prompts: Read and understand the user's prompt.
2. Implement Changes: Review and adjust current bounding boxes to meet user specifications.
3. Explain Adjustments: Justify the reasons behind each alteration and ensure every adjustment abides by the key guidelines.
4. Output the Result: Present the reasoning first, followed by the updated objects section, which should include a list of bounding boxes in Python format.

## Examples

- Example 1
    User prompt: A realistic image of landscape scene depicting a green car parking on the left of a blue truck, with a red air balloon and a bird in the sky
    Current Objects: [('green car #1', [0.027, 0.365, 0.275, 0.207]), ('blue truck #1', [0.350, 0.368, 0.272, 0.208]), ('red air balloon #1', [0.086, 0.010, 0.189, 0.176])]
    Reasoning: To add a bird in the sky as per the prompt, ensuring all coordinates and dimensions remain within [0, 1].
    Updated Objects: [('green car #1', [0.027, 0.365, 0.275, 0.207]), ('blue truck #1', [0.350, 0.369, 0.272, 0.208]), ('red air balloon #1', [0.086, 0.010, 0.189, 0.176]), ('bird #1', [0.385, 0.054, 0.186, 0.130])]

- Example 2
    User prompt: A realistic image of landscape scene depicting a green car parking on the right of a blue truck, with a red air balloon and a bird in the sky
    Current Output Objects: [('green car #1', [0.027, 0.365, 0.275, 0.207]), ('blue truck #1', [0.350, 0.369, 0.272, 0.208]), ('red air balloon #1', [0.086, 0.010, 0.189, 0.176])]
    Reasoning: The relative positions of the green car and blue truck do not match the prompt. Swap positions of the green car and blue truck to match the prompt, while keeping all coordinates and dimensions within [0, 1].
    Updated Objects:  [('green car #1', [0.350, 0.369, 0.275, 0.207]), ('blue truck #1', [0.027, 0.365, 0.272, 0.208]), ('red air balloon #1', [0.086, 0.010, 0.189, 0.176]), ('bird #1', [0.485, 0.054, 0.186, 0.130])]

- Example 3
    User prompt: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
    Current Objects: [('steam boat #1', [0.302, 0.293, 0.335, 0.194]), ('pink dolphin #1', [0.027, 0.324, 0.246, 0.160]), ('blue dolphin #1', [0.158, 0.454, 0.376, 0.290])]
    Reasoning: The prompt mentions only one dolphin, but two are present. Thus, remove one dolphin to match the prompt, ensuring all coordinates and dimensions stay within [0, 1].
    Updated Objects: [('steam boat #1', [0.302, 0.293, 0.335, 0.194]), ('pink dolphin #1', [0.027, 0.324, 0.246, 0.160])]

- Example 4
    User prompt: An oil painting of a pink dolphin jumping on the left of a steam boat on the sea
    Current Objects: [('steam boat #1', [0.302, 0.293, 0.335, 0.194]), ('dolphin #1', [0.027, 0.324, 0.246, 0.160])]
    Reasoning: The prompt specifies a pink dolphin, but there's only a generic one. The attribute needs to be changed.
    Updated Objects: [('steam boat #1', [0.302, 0.293, 0.335, 0.194]), ('pink dolphin #1', [0.027, 0.324, 0.246, 0.160])]

- Example 5
    User prompt: A realistic photo of a scene with a brown bowl on the right and a gray dog on the left
    Current Objects: [('gray dog #1', [0.186, 0.592, 0.449, 0.408]), ('brown bowl #1', [0.376, 0.194, 0.624, 0.502])]
    Reasoning: The leftmost coordinate (0.186) of the gray dog's bounding box is positioned to the left of the leftmost coordinate (0.376) of the brown bowl, while the rightmost coordinate (0.186 + 0.449) of the bounding box has not extended beyond the rightmost coordinate of the bowl. Thus, the image aligns with the user's prompt, requiring no further modifications.
    Updated Objects: [('gray dog #1', [0.186, 0.592, 0.449, 0.408]), ('brown bowl #1', [0.376, 0.194, 0.624, 0.502])]

Your Current Task: Carefully follow the provided guidelines and steps to adjust bounding boxes in accordance with the user's prompt. Ensure adherence to the above output format.

"""