from torchvision import transforms

"""
This module provides a set of common image transformation pipelines
for training and testing machine learning models using the torchvision library.

Image Processing / Augmentation Pipelines for IR Images
Input PIL Images 
Output torch Tensors [-1,1]    
"""


# IMAGE resolution
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512      # default, can be overridden by configration 

# Basic Transformation Pipeline
transform_pipeline = transforms.Compose ([ 
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),      # 10 Degrees Rotation 
    transforms.ToTensor(),
    transforms.Normalize([0.5,], [0.5,])    # normalize to [-1,1]
    
    ])

# Optional Augmentation Pipeline
augment_pipleine = transforms.Compose ([ 
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),      # 10 Degrees Rotation
    transforms.ToTensor(),
    transforms.Normalize([0.5,], [0.5,])    # normalize to [-1,1]
    ])
