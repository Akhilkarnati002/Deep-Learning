import numpy as np
from pathlib import Path
from PIL import Image


# ---------- utilities ----------

def largest_segment(mask):
    """
    Given a 1D boolean array, return (start, end) of the longest True segment.
    end is exclusive.
    """
    best_start = best_end = None
    current_start = None

    for i, v in enumerate(mask):
        if v and current_start is None:
            current_start = i
        elif (not v) and (current_start is not None):
            if best_start is None or (i - current_start) > (best_end - best_start):
                best_start, best_end = current_start, i
            current_start = None

    if current_start is not None:
        i = len(mask)
        if best_start is None or (i - current_start) > (best_end - best_start):
            best_start, best_end = current_start, i

    return best_start, best_end


def find_vertical_sliver(gray, frac=0.6, min_width=28, margin=3):
    """
    gray: 2D numpy array (H, W)
    Returns (x0, x1) columns of brightest vertical band.

    frac: threshold relative to max column brightness.
    """
    H, W = gray.shape
    col_profile = gray.mean(axis=0)

    if col_profile.max() == 0:
        return 0, W

    col_profile = col_profile / col_profile.max()
    mask = col_profile > frac

    if not mask.any():
        return 0, W

    start, end = largest_segment(mask)
    if start is None:
        return 0, W

    # enforce minimum width
    if (end - start) < min_width:
        center = (start + end) // 2
        start = max(0, center - min_width // 2)
        end = min(W, center + min_width // 2)

    # small margin on both sides
    start = max(0, start - margin)
    end = min(W, end + margin)
    return start, end


def find_horizontal_sliver(gray, frac=0.7, min_height=30, margin=3):
    """
    gray: 2D numpy array (H, W)
    Returns (y0, y1) rows of brightest horizontal band.

    Used for the high-res images to isolate the middle strip with the two circles.
    """
    H, W = gray.shape
    row_profile = gray.mean(axis=1)

    if row_profile.max() == 0:
        return 0, H

    row_profile = row_profile / row_profile.max()
    mask = row_profile > frac

    if not mask.any():
        return 0, H

    start, end = largest_segment(mask)
    if start is None:
        return 0, H

    # enforce minimum height
    if (end - start) < min_height:
        center = (start + end) // 2
        start = max(0, center - min_height // 2)
        end = min(H, center + min_height // 2)

    # small margin top/bottom
    start = max(0, start - margin)
    end = min(H, end + margin)
    return start, end


# ---------- cropping functions for each domain ----------

def crop_low_res(img, frac_vertical=0.7):
    """
    Low-res 382x288:
    - keep only the bright central vertical strip (drop magenta left/right)
    - keep full height
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    H, W = gray.shape

    x0, x1 = find_vertical_sliver(
        gray,
        frac=frac_vertical,          # tighter threshold; central strip is much brighter
        min_width=int(W * 0.1),      # at least 10% of width
        margin=3,
    )

    crop = img.crop((x0, 0, x1, H))
    return crop


def crop_high_res(img,
                  frac_vertical=0.5,
                  frac_horizontal=0.7):
    """
    High-res 174x512:
    - remove left blue area and right temperature scale
      via vertical brightness profile
    - then crop middle horizontal band with the two circles
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    H, W = gray.shape

    # center vertical strip without colorbar
    x0, x1 = find_vertical_sliver(
        gray,
        frac=frac_vertical,          # lower than low-res to allow central band
        min_width=int(W * 0.3),
        margin=3,
    )

    # now restrict to that vertical strip for horizontal profiling
    gray_strip = gray[:, x0:x1]
    y0, y1 = find_horizontal_sliver(
        gray_strip,
        frac=frac_horizontal,
        min_height=int(H * 0.25),
        margin=3,
    )

    crop = img.crop((x0, y0, x1, y1))
    return crop


# ---------- main driver ----------

def preprocess_all(
    in_dir_low,
    in_dir_high,
    out_dir_low,
    out_dir_high,
    target_size=256,
):
    in_dir_low = Path(in_dir_low)
    in_dir_high = Path(in_dir_high)
    out_dir_low = Path(out_dir_low)
    out_dir_high = Path(out_dir_high)

    out_dir_low.mkdir(parents=True, exist_ok=True)
    out_dir_high.mkdir(parents=True, exist_ok=True)

    allowed_ext = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

    # --- low-res ---
    for path in sorted(in_dir_low.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed_ext:
            continue

        print(f"[LOW] {path.name}")
        img = Image.open(path)

        cropped = crop_low_res(img)

        # resize to a square patch for training
        cropped = cropped.resize((target_size, target_size), Image.BILINEAR)

        out_path = out_dir_low / f"{path.stem}_crop.png"
        cropped.save(out_path)

    # --- high-res ---
    for path in sorted(in_dir_high.iterdir()):
        if not path.is_file() or path.suffix.lower() not in allowed_ext:
            continue

        print(f"[HIGH] {path.name}")
        img = Image.open(path)

        cropped = crop_high_res(img)

        cropped = cropped.resize((target_size, target_size), Image.BILINEAR)

        out_path = out_dir_high / f"{path.stem}_crop.png"
        cropped.save(out_path)


if __name__ == "__main__":
    preprocess_all(
        in_dir_high="Data/High_resolution/CFRP_60_high",
        in_dir_low="Data/Low_resolution/CFRP_60_low",
        out_dir_low="Data/low_cropped",
        out_dir_high="Data/high_cropped",
        target_size=256,
    )
