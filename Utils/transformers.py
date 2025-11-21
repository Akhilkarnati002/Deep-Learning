from torchvision import transforms

"""
This module provides a set of common image transformation pipelines
for training and testing machine learning models using the torchvision library.

Image Processing / Augmentation Pipelines for IR Images
Input PIL Images
Output torch Tensors in [-1, 1]
"""


# IMAGE resolution
IMAGE_HEIGHT = 256  # default, can be overridden by configuration
IMAGE_WIDTH = 512   # default, can be overridden by configuration

# Basic Transformation Pipeline
# For paired low/high images it is important that both undergo the same
# spatial transformations. To keep things simple and deterministic here,
# we only use resize + normalization (no random flip/rotation).
transform_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # normalize to [-1,1]
])

# Optional Augmentation Pipeline (unused in current training loop)
augment_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
