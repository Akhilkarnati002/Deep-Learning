from torchvision import transforms

IMAGE_SIZE = 256

transform_pipeline = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])   # map to [-1, 1]
])
