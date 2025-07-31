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
def process_image(input_path, output_path, size=(1024, 1024), scaling_mode='pad'):
    """
    Process an image with high-quality scaling.

    Args:
        input_path: Path to input image
        output_path: Path to save output image
        size: Target size as (width, height) or single number for square
        scaling_mode: Either 'pad' (preserve aspect + pad) or 'scale' (direct scale)
    """
    ext = input_path.suffix.lower()
    if ext in [".jpg", ".png"]:
        with Image.open(input_path) as im:
            im = im.convert("RGB")

            # Handle size parameter
            if isinstance(size, int):
                size = (size, size)

            if scaling_mode == 'scale':
                # Direct scaling to target size with high-quality filter
                im_resized = im.resize(size, resample=Image.Resampling.LANCZOS)
                im_resized.save(output_path, format="PNG")
            else:  # pad mode
                w, h = im.size
                max_size = max(size)

                # Compute scale while preserving aspect
                scale = max_size / float(max(w, h))
                new_w = int(w * scale)
                new_h = int(h * scale)

                # High-quality resize
                im_resized = im.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

                # Create canvas and paste
                bg = Image.new("RGB", (max_size, max_size))
                offset_x = (max_size - new_w) // 2
                offset_y = (max_size - new_h) // 2
                bg.paste(im_resized, (offset_x, offset_y))

                # Fill padding with edge replication
                def get_resized_pixel(x, y):
                    if x < 0: x = 0
                    elif x >= new_w: x = new_w - 1
                    if y < 0: y = 0
                    elif y >= new_h: y = new_h - 1
                    return im_resized.getpixel((x, y))

                # Fill top & bottom
                for y in range(0, offset_y):
                    for x in range(max_size):
                        x_in_resized = x - offset_x
                        bg.putpixel((x, y), get_resized_pixel(x_in_resized, 0))

                for y in range(offset_y + new_h, max_size):
                    for x in range(max_size):
                        x_in_resized = x - offset_x
                        bg.putpixel((x, y), get_resized_pixel(x_in_resized, new_h - 1))

                # Fill left & right
                for y in range(offset_y, offset_y + new_h):
                    for x in range(0, offset_x):
                        y_in_resized = y - offset_y
                        bg.putpixel((x, y), get_resized_pixel(0, y_in_resized))

                    for x in range(offset_x + new_w, max_size):
                        y_in_resized = y - offset_y
                        bg.putpixel((x, y), get_resized_pixel(new_w - 1, y_in_resized))

                bg.save(output_path, format="PNG")
    elif ext == ".exr":
        process_exr(input_path, output_path, size)

# Function to convert a single file with specific output name
def convert_single_file(input_path, output_path, size=(1024, 1024), scaling_mode='pad'):
    input_path = Path(input_path)
    output_path = Path(output_path)
    os.makedirs(output_path.parent, exist_ok=True)
    process_image(input_path, output_path, size=size, scaling_mode=scaling_mode)

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
        elif "_BaseColor" in file_name:
            base_name = file_name.split("_BaseColor")[0]
            output_name = f"{base_name}_albedo.png"
        elif "_Normal" in file_name:
            base_name = file_name.split("_Normal")[0]
            output_name = f"{base_name}_normal.png"
        elif "_Roughness" in file_name:
            base_name = file_name.split("_Roughness")[0]
            output_name = f"{base_name}_rough.png"
        elif "_BaseColor" in file_name:
            base_name = file_name.split("_BaseColor")[0]
            output_name = f"{base_name}_albedo.png"
        elif "_Normal" in file_name:
            base_name = file_name.split("_Normal")[0]
            output_name = f"{base_name}_normal.png"
        elif "_albedo" in file_name:
            base_name = file_name.split("_albedo")[0]
            output_name = f"{base_name}_albedo.png"
        elif "-Metallic" in file_name:
            base_name = file_name.split("-Metallic")[0]
            output_name = f"{base_name}_metal.png"
        elif "-albedo" in file_name:
            base_name = file_name.split("-albedo")[0]
            output_name = f"{base_name}_albedo.png"
        elif "-Roughness" in file_name:
            base_name = file_name.split("-Roughness")[0]
            output_name = f"{base_name}_rough.png"
        elif "-Normal" in file_name:
            base_name = file_name.split("-Normal")[0]
            output_name = f"{base_name}_normal.png"

        process_image(file_path, output_folder / output_name)

# Example usage
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        # Get size and mode from args if provided
        size = 1024  # default
        scaling_mode = 'pad'  # default
        if len(sys.argv) > 3:
            size = int(sys.argv[3])
        if len(sys.argv) > 4:
            scaling_mode = sys.argv[4]
        convert_single_file(sys.argv[1], sys.argv[2], size=size, scaling_mode=scaling_mode)
    else:
        # Default folder conversion
        input_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/input")
        output_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/output")
        convert_images(input_folder, output_folder)