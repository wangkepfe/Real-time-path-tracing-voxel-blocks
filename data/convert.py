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
        """
        1) Reads an image from input_path, converts to RGB.
        2) Finds a scale factor so that the max dimension becomes `max_size`.
        3) Resizes (downscales) the image preserving aspect ratio.
        4) Pastes this resized image into a new (max_size x max_size) canvas.
        5) Fills the padding region by *replicating* the nearest edge pixels.
        6) Saves as PNG to output_path.
        """

        with Image.open(input_path) as im:
            im = im.convert("RGB")
            w, h = im.size

            max_size = 1024

            # -------------------------------------------------------
            # 1) Compute scale so that the largest side is `max_size`
            # -------------------------------------------------------
            scale = max_size / float(max(w, h))
            new_w = int(w * scale)
            new_h = int(h * scale)

            # --------------------------------
            # 2) Resize while preserving aspect
            # --------------------------------
            im_resized = im.resize((new_w, new_h), resample=Image.LANCZOS)

            # ----------------------------------------------------
            # 3) Create a new blank (1024x1024) canvas, paste image
            # ----------------------------------------------------
            bg = Image.new("RGB", (max_size, max_size))
            # Centering offset so that the resized image is centered
            offset_x = (max_size - new_w) // 2
            offset_y = (max_size - new_h) // 2
            bg.paste(im_resized, (offset_x, offset_y))

            # ----------------------------------------------------
            # 4) Replicate border pixels to fill the padded region
            # ----------------------------------------------------

            # Helper: safely get a pixel from im_resized at (x, y),
            # clamped in case x or y is out of range:
            def get_resized_pixel(x, y):
                # clamp x
                if x < 0:
                    x = 0
                elif x >= new_w:
                    x = new_w - 1
                # clamp y
                if y < 0:
                    y = 0
                elif y >= new_h:
                    y = new_h - 1
                return im_resized.getpixel((x, y))

            # Fill top & bottom (rows outside offset_y .. offset_y+new_h)
            for y in range(0, offset_y):
                for x in range(max_size):
                    # vertical replicate of top row => nearest row in resized image is y=0
                    # but we must translate x into the local coordinate in im_resized
                    # x_in_resized = x - offset_x
                    x_in_resized = x - offset_x
                    bg.putpixel((x, y), get_resized_pixel(x_in_resized, 0))

            for y in range(offset_y + new_h, max_size):
                for x in range(max_size):
                    x_in_resized = x - offset_x
                    bg.putpixel((x, y), get_resized_pixel(x_in_resized, new_h - 1))

            # Fill left & right (columns outside offset_x .. offset_x+new_w)
            # But only in the vertical region that corresponds to the actual resized image
            for y in range(offset_y, offset_y + new_h):
                for x in range(0, offset_x):
                    # horizontal replicate of left column => nearest col in resized image is x=0
                    y_in_resized = y - offset_y
                    bg.putpixel((x, y), get_resized_pixel(0, y_in_resized))

                for x in range(offset_x + new_w, max_size):
                    y_in_resized = y - offset_y
                    bg.putpixel((x, y), get_resized_pixel(new_w - 1, y_in_resized))

            # ----------------------------------------------------
            # 5) Save as PNG
            # ----------------------------------------------------
            bg.save(output_path, format="PNG")
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
    input_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/input")
    output_folder = Path("D:/cmder/Real-time-path-tracing-voxel-blocks/data/output")

    convert_images(input_folder, output_folder)