#!/usr/bin/env python3
"""
High-Fidelity Texture Upscaler

Upscales the pink-smoothie.png texture from 64x64 to 4096x4096 by replacing each 1x1 texel
with a semantically-appropriate 64x64 texture patch from the generated textures.
"""

import numpy as np
from PIL import Image, ImageColor
import json
import os
import math
from typing import Dict, Tuple, Optional

class TextureUpscaler:
    def __init__(self, upscale_factor=64, texture_size=512):
        self.upscale_factor = upscale_factor
        self.texture_size = texture_size
        self.generated_textures_dir = "generated_textures"

        # Load semantic mapping
        with open("pink-smoothie-semantic.json", 'r') as f:
            self.semantic_data = json.load(f)

        self.labels = self.semantic_data['labels']
        self.semantic_map = np.array(self.semantic_data['semantic_map'])

        # Material to texture mapping
        self.material_mapping = {
            '0': None,  # None/transparent
            '1': 'human_skin',      # Skin
            '2': 'hair',            # Hair
            '4': 'shiny_silver_metal',  # Metal
            '5': 'leather',         # Leather
            '6': 'human_skin',      # Eyes (use skin texture)
            '7': 'ribbon',          # Ribbon
            '8': 'shiny_black_leather',  # Shiny-Leather
            '9': 'pink_terry_cloth',     # Fabric1 (Pink cotton terry cloth)
            '10': 'white_cotton',        # Fabric2 (Natural cotton plaid)
            '11': 'white_cotton'         # Fabric3 (White cotton)
        }

        # Load texture cache
        self.texture_cache = {}
        self._load_textures()

    def _load_textures(self):
        """Load all available textures into memory"""
        print("Loading texture cache...")
        for material in set(self.material_mapping.values()):
            if material is not None:
                self.texture_cache[material] = {}
                for texture_type in ['albedo', 'normal', 'roughness']:
                    texture_path = os.path.join(self.generated_textures_dir, f"{material}_{texture_type}.png")
                    if os.path.exists(texture_path):
                        self.texture_cache[material][texture_type] = np.array(Image.open(texture_path))
                        print(f"  Loaded {material}_{texture_type}")

    def _get_texture_crop(self, material: str, texture_type: str, x_offset: int, y_offset: int) -> np.ndarray:
        """Get a 64x64 crop from a 512x512 texture with given offset for continuity"""
        if material not in self.texture_cache or texture_type not in self.texture_cache[material]:
            # Return a solid color if texture not found
            color = (128, 128, 128) if texture_type == 'albedo' else (128, 128, 255) if texture_type == 'normal' else (128,)
            channels = len(color)
            return np.full((self.upscale_factor, self.upscale_factor, channels), color, dtype=np.uint8)

        texture = self.texture_cache[material][texture_type]

        # Calculate crop coordinates with wrapping for seamless textures
        start_x = x_offset % self.texture_size
        start_y = y_offset % self.texture_size

                # Handle wrapping around texture boundaries
        # Handle both grayscale and color textures
        if len(texture.shape) == 2:
            # Grayscale texture
            crop = np.zeros((self.upscale_factor, self.upscale_factor), dtype=np.uint8)
            for i in range(self.upscale_factor):
                for j in range(self.upscale_factor):
                    src_x = (start_x + j) % self.texture_size
                    src_y = (start_y + i) % self.texture_size
                    crop[i, j] = texture[src_y, src_x]
            # Add channel dimension for consistency
            crop = crop[:, :, np.newaxis]
        else:
            # Color texture
            crop = np.zeros((self.upscale_factor, self.upscale_factor, texture.shape[2]), dtype=np.uint8)
            for i in range(self.upscale_factor):
                for j in range(self.upscale_factor):
                    src_x = (start_x + j) % self.texture_size
                    src_y = (start_y + i) % self.texture_size
                    crop[i, j] = texture[src_y, src_x]

        return crop

    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Convert RGB to HSV"""
        r, g, b = [x / 255.0 for x in rgb]
        max_val = max(r, g, b)
        min_val = min(r, g, b)
        diff = max_val - min_val

        # Hue calculation
        if diff == 0:
            h = 0
        elif max_val == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_val == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360

        # Saturation calculation
        s = 0 if max_val == 0 else diff / max_val

        # Value calculation
        v = max_val

        return h, s, v

    def _hsv_to_rgb(self, hsv: Tuple[float, float, float]) -> Tuple[int, int, int]:
        """Convert HSV to RGB"""
        h, s, v = hsv
        c = v * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = v - c

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))

    def _adjust_color(self, texture_crop: np.ndarray, target_color: Tuple[int, int, int]) -> np.ndarray:
        """Adjust the color of a texture crop to match the target color"""
        if len(texture_crop.shape) != 3 or texture_crop.shape[2] < 3:
            return texture_crop

        # Convert target color to HSV
        target_h, target_s, target_v = self._rgb_to_hsv(target_color)

        # Calculate average color of the texture crop
        avg_color = np.mean(texture_crop[:, :, :3], axis=(0, 1))
        crop_h, crop_s, crop_v = self._rgb_to_hsv(tuple(avg_color.astype(int)))

        # Adjust the texture crop
        adjusted_crop = texture_crop.copy()

        # Convert each pixel to HSV, adjust, and convert back
        for i in range(texture_crop.shape[0]):
            for j in range(texture_crop.shape[1]):
                pixel_rgb = tuple(texture_crop[i, j, :3])
                pixel_h, pixel_s, pixel_v = self._rgb_to_hsv(pixel_rgb)

                # Adjust hue and saturation to match target, preserve relative value changes
                adjusted_h = target_h
                adjusted_s = target_s
                adjusted_v = target_v * (pixel_v / max(crop_v, 0.001))

                # Convert back to RGB
                adjusted_rgb = self._hsv_to_rgb((adjusted_h, adjusted_s, adjusted_v))
                # Clamp values to 0-255 range
                adjusted_rgb = tuple(max(0, min(255, int(val))) for val in adjusted_rgb)
                adjusted_crop[i, j, :3] = adjusted_rgb

        return adjusted_crop

    def _calculate_texture_offset(self, x: int, y: int, material: str) -> Tuple[int, int]:
        """Calculate texture offset to ensure continuity for adjacent texels with same material"""
        # Use position-based offset for continuity
        # Scale the offset to create seamless transitions
        offset_scale = self.upscale_factor
        x_offset = (x * offset_scale) % self.texture_size
        y_offset = (y * offset_scale) % self.texture_size
        return x_offset, y_offset

    def upscale_texture(self, input_image_path: str, output_path: str = "high_fidelity_pink_smoothie.png"):
        """Main function to upscale the texture"""
        print(f"Loading input image: {input_image_path}")

        # Load the original low-res texture
        original_image = Image.open(input_image_path).convert("RGBA")
        original_array = np.array(original_image)

        original_height, original_width = original_array.shape[:2]

        # Create output image
        output_width = original_width * self.upscale_factor
        output_height = original_height * self.upscale_factor
        output_array = np.zeros((output_height, output_width, 4), dtype=np.uint8)

        print(f"Upscaling from {original_width}x{original_height} to {output_width}x{output_height}")

        # Process each texel
        for y in range(original_height):
            for x in range(original_width):
                # Get original pixel
                original_pixel = original_array[y, x]
                alpha = original_pixel[3]

                # Calculate output position
                out_y_start = y * self.upscale_factor
                out_x_start = x * self.upscale_factor
                out_y_end = out_y_start + self.upscale_factor
                out_x_end = out_x_start + self.upscale_factor

                if alpha == 0:
                    # Transparent pixel - keep transparent
                    output_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = 0
                    continue

                # Get semantic label
                if y < len(self.semantic_map) and x < len(self.semantic_map[0]):
                    label = str(self.semantic_map[y][x])
                else:
                    label = '0'  # Default to None if out of bounds

                # Get material for this label
                material = self.material_mapping.get(label)

                if material is None:
                    # No material mapping, use original color
                    solid_color = original_pixel[:3]
                    output_array[out_y_start:out_y_end, out_x_start:out_x_end, :3] = solid_color
                    output_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = alpha
                else:
                    # Get texture offset for continuity
                    x_offset, y_offset = self._calculate_texture_offset(x, y, material)

                    # Get albedo texture crop
                    albedo_crop = self._get_texture_crop(material, 'albedo', x_offset, y_offset)

                    # Adjust color to match original
                    target_color = tuple(original_pixel[:3])
                    adjusted_crop = self._adjust_color(albedo_crop, target_color)

                    # Copy to output
                    output_array[out_y_start:out_y_end, out_x_start:out_x_end, :3] = adjusted_crop[:, :, :3]
                    output_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = alpha

                # Progress indicator
                if (y * original_width + x) % (original_width * 4) == 0:
                    progress = ((y * original_width + x) / (original_width * original_height)) * 100
                    print(f"Progress: {progress:.1f}%")

        # Save output
        output_image = Image.fromarray(output_array, 'RGBA')
        output_image.save(output_path)
        print(f"High-fidelity texture saved to: {output_path}")

        # Also save individual texture maps if available
        self._save_additional_maps(input_image_path, output_path)

    def _save_additional_maps(self, input_image_path: str, output_path: str):
        """Generate and save normal and roughness maps"""
        print("Generating normal and roughness maps...")

        # Load the original low-res texture
        original_image = Image.open(input_image_path).convert("RGBA")
        original_array = np.array(original_image)
        original_height, original_width = original_array.shape[:2]

        # Create output arrays for normal and roughness
        output_width = original_width * self.upscale_factor
        output_height = original_height * self.upscale_factor

        normal_array = np.full((output_height, output_width, 3), (128, 128, 255), dtype=np.uint8)
        roughness_array = np.full((output_height, output_width, 1), 128, dtype=np.uint8)

        # Process each texel
        for y in range(original_height):
            for x in range(original_width):
                original_pixel = original_array[y, x]
                alpha = original_pixel[3]

                if alpha == 0:
                    continue

                # Calculate output position
                out_y_start = y * self.upscale_factor
                out_x_start = x * self.upscale_factor
                out_y_end = out_y_start + self.upscale_factor
                out_x_end = out_x_start + self.upscale_factor

                # Get semantic label and material
                label = str(self.semantic_map[y][x]) if y < len(self.semantic_map) and x < len(self.semantic_map[0]) else '0'
                material = self.material_mapping.get(label)

                if material is not None:
                    # Get texture offset for continuity
                    x_offset, y_offset = self._calculate_texture_offset(x, y, material)

                    # Get normal and roughness crops
                    normal_crop = self._get_texture_crop(material, 'normal', x_offset, y_offset)
                    roughness_crop = self._get_texture_crop(material, 'roughness', x_offset, y_offset)

                    # Copy to output arrays
                    if normal_crop.shape[2] >= 3:
                        normal_array[out_y_start:out_y_end, out_x_start:out_x_end] = normal_crop[:, :, :3]
                    if roughness_crop.shape[2] >= 1:
                        roughness_array[out_y_start:out_y_end, out_x_start:out_x_end] = roughness_crop[:, :, :1]

        # Save additional maps
        base_name = output_path.rsplit('.', 1)[0]

        normal_image = Image.fromarray(normal_array, 'RGB')
        normal_image.save(f"{base_name}_normal.png")
        print(f"Normal map saved to: {base_name}_normal.png")

        roughness_image = Image.fromarray(roughness_array.squeeze(), 'L')
        roughness_image.save(f"{base_name}_roughness.png")
        print(f"Roughness map saved to: {base_name}_roughness.png")

def main():
    upscaler = TextureUpscaler()
    upscaler.upscale_texture("pink-smoothie.png")

if __name__ == "__main__":
    main()