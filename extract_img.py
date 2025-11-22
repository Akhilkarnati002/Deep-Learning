import cv2
import numpy as np
from PIL import Image

img = np.array(Image.open("./Data/Low_resolution/CFRP_60_low/Record_2025-11-11_10-42-17.tiff"))
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Option A: Otsu threshold
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Option B: percentile threshold (good for IR)
thresh_val = np.percentile(gray, 95)
mask = (gray >= thresh_val).astype(np.uint8) * 255

# Morphological cleanup (optional)
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
