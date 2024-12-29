import os
import glob
from pathlib import Path
from PIL import Image
import cv2
import OpenEXR
import Imath
import numpy as np

# Function to process EXR files
def process_exr(input_path, output_path, size=(1024, 1024)):
    exr_file = OpenEXR.InputFile(str(input_path))
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    header = exr_file.header()
    print("Channels:", list(header["channels"].keys()))
    for ch_name, ch_info in header["channels"].items():
        print(ch_name, ch_info.type, ch_info.type.v)

    # Determine if the EXR file is mono or RGB
    channels = exr_file.header()['channels'].keys()

    if 'R' in channels and 'G' in channels and 'B' in channels:
        # RGB image
        r, g, b = [
            np.frombuffer(exr_file.channel(c, Imath.PixelType(Imath.PixelType.HALF)), dtype=np.float16).reshape(height, width)
            for c in ('R', 'G', 'B')
        ]
        img = np.stack([b, g, r], axis=2)
    else:
        # Mono image (e.g., roughness)
        img = np.frombuffer(
            exr_file.channel('Y', Imath.PixelType(Imath.PixelType.HALF)), dtype=np.float16
        ).reshape(height, width)

    # Normalize and resize
    img = (img - img.min()) / (img.max() - img.min()) * 255  # Normalize to 0-255
    img = img.astype(np.uint8)
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    # Save as PNG
    cv2.imwrite(str(output_path), img_resized)

# Function to process a single image and save it with the new name and format
def process_image(input_path, output_path, size=(1024, 1024)):
    ext = input_path.suffix.lower()
    if ext in [".jpg", ".png"]:
        image = Image.open(input_path).convert("RGB")
        image = image.resize(size, Image.LANCZOS)
        image.save(output_path, format="PNG")
    elif ext == ".exr":
        process_exr(input_path, output_path, size)

# Main function to process folders
def convert_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        file_path = input_folder / file_name

        print(file_name)

        if "_diff" in file_name:
            base_name = file_name.split("_diff")[0]
            output_name = f"{base_name}_albedo.png"
        elif "_nor" in file_name:
            base_name = file_name.split("_nor")[0]
            output_name = f"{base_name}_normal.png"
        elif "_rough" in file_name:
            base_name = file_name.split("_rough")[0]
            output_name = f"{base_name}_rough.png"

        process_image(file_path, output_folder / output_name)

# Example usage
if __name__ == "__main__":
    input_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/input")
    output_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/output")

    convert_images(input_folder, output_folder)