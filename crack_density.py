import os
import glob
import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
from skimage import io
import matplotlib.pyplot as plt

INPUT_DIR = r"your_input_folder"
OUTPUT_DIR = r"your_output_folder"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PIXEL_TO_UM = 500 / 450

def process_and_save(img_path):
    name = os.path.basename(img_path).split(".")[0]
    print(f"Processing: {name}")

    img = io.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = gray.astype(np.float32)

    # Background removal
    bg = cv2.GaussianBlur(g, (101, 101), 0)
    hp = g - bg
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # CLAHE enhancement
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enh = clahe.apply(hp)

    # Black-hat for crack extraction
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(enh, cv2.MORPH_BLACKHAT, kernel)

    # Thresholding
    t = np.percentile(blackhat, 95)
    bw = blackhat > t

    # Noise removal
    bw = binary_closing(bw, disk(1))
    bw = remove_small_objects(bw, min_size=200)

    # Skeletonization
    sk = skeletonize(bw)

    # Crack density
    crack_len = np.sum(sk)
    area = sk.shape[0] * sk.shape[1]
    rho = (crack_len / area) / PIXEL_TO_UM

    print(f"{name} density = {rho:.6f}")

    # Visualization
    overlay = img.copy()
    overlay[sk] = [255, 0, 0]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(bw, cmap='gray'); plt.title("Binary"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(sk, cmap='gray'); plt.title("Skeleton"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(overlay); plt.title("Overlay"); plt.axis('off')
    plt.show()

    # Save
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_binary.png"), bw.astype(np.uint8)*255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_skeleton.png"), sk.astype(np.uint8)*255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return rho

files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

for f in files:
    process_and_save(f)