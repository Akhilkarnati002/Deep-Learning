from Utils.dataset import IRImageDataset
from Utils.transformers import transform_pipeline



low_dir = "Data/Low_resolution/CFRP_60_low"
high_dir = "Data/High_resolution/CFRP_60_high"


dataset = IRImageDataset(low_dir, high_dir, transform=transform_pipeline, paired=True)
print(dataset[0])


# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)