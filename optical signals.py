import cv2
import os
import numpy as np

# Functions to read and write images
def read_image(filepath):
    """Read an image in grayscale using OpenCV."""
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return img
    except Exception as e:
        print(f"Failed to read: {e}")
        return None

def write_image(filepath, img):
    """Write an image using OpenCV."""
    try:
        success = cv2.imwrite(filepath, img)
        return success
    except Exception as e:
        print(f"Failed to write: {e}")
        return False

# --- Paths (relative, GitHub-friendly) ---
input_folder = os.path.join("data", "input_images")           # Place input images here
binary_folder = os.path.join("data", "binary_images")         # Binary images will be saved here
bitstream_folder = os.path.join("data", "bitstreams")         # Bitstream files will be saved here

# Make sure output folders exist
os.makedirs(binary_folder, exist_ok=True)
os.makedirs(bitstream_folder, exist_ok=True)

# Standard size for output images
standard_size = (150, 150)

# Statistics counters
total_files = 0
success_files = 0
failed_files = 0

print("Starting image processing...\n")

# Iterate over all subfolders in the input folder
for subfolder_name in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder_name)

    # Skip output folders themselves (if accidentally inside input)
    if subfolder_name in ['binary_images', 'bitstreams']:
        continue

    if os.path.isdir(subfolder_path):
        print(f"Processing folder: {subfolder_name}")

        # Create corresponding output subfolders
        binary_subfolder = os.path.join(binary_folder, subfolder_name)
        bitstream_subfolder = os.path.join(bitstream_folder, subfolder_name)
        os.makedirs(binary_subfolder, exist_ok=True)
        os.makedirs(bitstream_subfolder, exist_ok=True)

        # Process each image in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                total_files += 1
                image_path = os.path.join(subfolder_path, filename)

                # Read image
                gray_image = read_image(image_path)
                if gray_image is None:
                    print(f" ✗ Failed to read: {filename}")
                    failed_files += 1
                    continue

                try:
                    # Resize image
                    resized = cv2.resize(gray_image, standard_size, interpolation=cv2.INTER_LINEAR)
                    # Apply Gaussian blur
                    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
                    # Adaptive thresholding
                    binary_image = cv2.adaptiveThreshold(
                        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 11, 2
                    )

                    # Save binary image
                    binary_output_path = os.path.join(binary_subfolder, filename)
                    if not write_image(binary_output_path, binary_image):
                        print(f" ✗ Failed to save: {filename}")
                        failed_files += 1
                        continue

                    # Flatten to 1D bitstream (0/1)
                    bitstream = (binary_image.flatten() // 255).astype(np.uint8)
                    # Save bitstream as text file
                    bitstream_filename = os.path.splitext(filename)[0] + "_bitstream.txt"
                    bitstream_path = os.path.join(bitstream_subfolder, bitstream_filename)
                    with open(bitstream_path, 'w') as f:
                        f.write(''.join(map(str, bitstream)))

                    success_files += 1
                    if success_files % 50 == 0:
                        print(f" ✔ Successfully processed {success_files} files...")

                except Exception as e:
                    print(f" ✗ Processing failed for {filename}: {str(e)}")
                    failed_files += 1

# Print statistics
print(f"\n{'='*60}")
print("Processing complete!")
print(f"Total files: {total_files}")
print(f"✔ Successful: {success_files}")
print(f"✗ Failed: {failed_files}")
print(f"Success rate: {success_files / total_files * 100:.1f}%" if total_files > 0 else "N/A")
print(f"{'='*60}")
print(f"✔ Binary images saved to: {binary_folder}")
print(f"✔ Bitstream files saved to: {bitstream_folder}")
