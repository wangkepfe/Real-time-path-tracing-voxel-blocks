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
import argparse
from typing import Dict, Tuple, Optional

class TextureUpscaler:
    def __init__(self, semantic_file, material_mapping_file, generated_textures_dir, upscale_factor=64, texture_size=512, source_texture_scale=1.0):
        self.upscale_factor = upscale_factor
        self.texture_size = texture_size
        self.source_texture_scale = source_texture_scale
        self.generated_textures_dir = generated_textures_dir

        # Load semantic mapping
        with open(semantic_file, 'r') as f:
            self.semantic_data = json.load(f)

        self.labels = self.semantic_data['labels']
        self.semantic_map = np.array(self.semantic_data['semantic_map'])

        # Load material mapping from file
        self.material_mapping = self._load_material_mapping(material_mapping_file)

        # Load texture cache
        self.texture_cache = {}
        self._load_textures()

    def _load_material_mapping(self, mapping_file: str) -> Dict[str, Optional[str]]:
        """Load material mapping from JSON file"""
        with open(mapping_file, 'r') as f:
            mapping_data = json.load(f)
        print(f"Loaded material mapping from {mapping_file}")
        return mapping_data

    def _load_textures(self):
        """Load all available textures into memory"""
        print("Loading texture cache...")
        for material in set(self.material_mapping.values()):
            if material is not None:
                self.texture_cache[material] = {}
                for texture_type in ['albedo', 'normal', 'roughness']:
                    texture_path = os.path.join(self.generated_textures_dir, f"{material}_{texture_type}.png")
                    if os.path.exists(texture_path):
                        img = Image.open(texture_path)
                        
                        # Scale the source texture if needed
                        if self.source_texture_scale != 1.0:
                            original_size = img.size
                            new_size = (int(original_size[0] * self.source_texture_scale), 
                                      int(original_size[1] * self.source_texture_scale))
                            img = img.resize(new_size, Image.NEAREST)
                            print(f"  Scaled {material}_{texture_type} from {original_size} to {new_size}")
                        
                        self.texture_cache[material][texture_type] = np.array(img)
                        print(f"  Loaded {material}_{texture_type}")

    def _get_texture_crop(self, material: str, texture_type: str, x_offset: int, y_offset: int) -> np.ndarray:
        """Get a crop from texture with given offset for continuity"""
        if material not in self.texture_cache or texture_type not in self.texture_cache[material]:
            # Return a solid color if texture not found
            color = (128, 128, 128) if texture_type == 'albedo' else (128, 128, 255) if texture_type == 'normal' else (128,)
            channels = len(color)
            return np.full((self.upscale_factor, self.upscale_factor, channels), color, dtype=np.uint8)

        texture = self.texture_cache[material][texture_type]
        
        # Get actual texture size (may be scaled)
        actual_texture_size = texture.shape[0] if len(texture.shape) >= 2 else self.texture_size

        # Calculate crop coordinates with wrapping for seamless textures
        start_x = x_offset % actual_texture_size
        start_y = y_offset % actual_texture_size

        # Handle wrapping around texture boundaries
        # Handle both grayscale and color textures
        if len(texture.shape) == 2:
            # Grayscale texture
            crop = np.zeros((self.upscale_factor, self.upscale_factor), dtype=np.uint8)
            for i in range(self.upscale_factor):
                for j in range(self.upscale_factor):
                    src_x = (start_x + j) % actual_texture_size
                    src_y = (start_y + i) % actual_texture_size
                    crop[i, j] = texture[src_y, src_x]
            # Add channel dimension for consistency
            crop = crop[:, :, np.newaxis]
        else:
            # Color texture
            crop = np.zeros((self.upscale_factor, self.upscale_factor, texture.shape[2]), dtype=np.uint8)
            for i in range(self.upscale_factor):
                for j in range(self.upscale_factor):
                    src_x = (start_x + j) % actual_texture_size
                    src_y = (start_y + i) % actual_texture_size
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
        # Get actual texture size for this material (may be scaled)
        actual_texture_size = self.texture_size
        if (material in self.texture_cache and 'albedo' in self.texture_cache[material]):
            texture = self.texture_cache[material]['albedo']
            actual_texture_size = texture.shape[0] if len(texture.shape) >= 2 else self.texture_size
        
        # Use position-based offset for continuity
        # Scale the offset to create seamless transitions
        offset_scale = self.upscale_factor
        x_offset = (x * offset_scale) % actual_texture_size
        y_offset = (y * offset_scale) % actual_texture_size
        return x_offset, y_offset

    def upscale_texture(self, input_image_path: str, output_base_name: str):
        """Main function to upscale the texture"""
        print(f"Loading input image: {input_image_path}")

        # Generate output paths
        albedo_path = f"{output_base_name}_albedo.png"
        normal_path = f"{output_base_name}_normal.png"
        roughness_path = f"{output_base_name}_roughness.png"

        # Load the original low-res texture
        original_image = Image.open(input_image_path).convert("RGBA")
        original_array = np.array(original_image)

        original_height, original_width = original_array.shape[:2]

        # Create output arrays
        output_width = original_width * self.upscale_factor
        output_height = original_height * self.upscale_factor
        
        albedo_array = np.zeros((output_height, output_width, 4), dtype=np.uint8)
        normal_array = np.full((output_height, output_width, 3), (128, 128, 255), dtype=np.uint8)
        roughness_array = np.full((output_height, output_width, 1), 128, dtype=np.uint8)

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
                    albedo_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = 0
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
                    albedo_array[out_y_start:out_y_end, out_x_start:out_x_end, :3] = solid_color
                    albedo_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = alpha
                else:
                    # Get texture offset for continuity
                    x_offset, y_offset = self._calculate_texture_offset(x, y, material)

                    # Get texture crops for all types
                    albedo_crop = self._get_texture_crop(material, 'albedo', x_offset, y_offset)
                    normal_crop = self._get_texture_crop(material, 'normal', x_offset, y_offset)
                    roughness_crop = self._get_texture_crop(material, 'roughness', x_offset, y_offset)

                    # Adjust albedo color to match original
                    target_color = tuple(original_pixel[:3])
                    adjusted_albedo = self._adjust_color(albedo_crop, target_color)

                    # Copy to output arrays
                    albedo_array[out_y_start:out_y_end, out_x_start:out_x_end, :3] = adjusted_albedo[:, :, :3]
                    albedo_array[out_y_start:out_y_end, out_x_start:out_x_end, 3] = alpha
                    
                    if normal_crop.shape[2] >= 3:
                        normal_array[out_y_start:out_y_end, out_x_start:out_x_end] = normal_crop[:, :, :3]
                    if roughness_crop.shape[2] >= 1:
                        roughness_array[out_y_start:out_y_end, out_x_start:out_x_end] = roughness_crop[:, :, :1]

                # Progress indicator
                if (y * original_width + x) % (original_width * 4) == 0:
                    progress = ((y * original_width + x) / (original_width * original_height)) * 100
                    print(f"Progress: {progress:.1f}%")

        # Save all output images
        albedo_image = Image.fromarray(albedo_array, 'RGBA')
        albedo_image.save(albedo_path)
        print(f"Albedo texture saved to: {albedo_path}")

        normal_image = Image.fromarray(normal_array, 'RGB')
        normal_image.save(normal_path)
        print(f"Normal map saved to: {normal_path}")

        roughness_image = Image.fromarray(roughness_array.squeeze(), 'L')
        roughness_image.save(roughness_path)
        print(f"Roughness map saved to: {roughness_path}")


def main():
    parser = argparse.ArgumentParser(description='High-Fidelity Texture Upscaler')
    parser.add_argument('input', help='Input texture file path')
    parser.add_argument('semantic', help='Semantic mapping JSON file path')
    parser.add_argument('material_mapping', help='Material mapping JSON file path')
    parser.add_argument('generated_textures_dir', help='Directory containing generated texture files')
    parser.add_argument('-o', '--output', help='Output base name (generates _albedo.png, _normal.png, _roughness.png files)')
    parser.add_argument('--upscale-factor', type=int, default=64, help='Upscale factor (default: 64)')
    parser.add_argument('--texture-size', type=int, default=512, help='Source texture size (default: 512)')
    parser.add_argument('--source-texture-scale', type=float, default=1.0, help='Scale factor for source textures before cropping (default: 1.0)')
    
    args = parser.parse_args()
    
    # Generate output base name if not provided
    if args.output is None:
        input_name, input_ext = os.path.splitext(args.input)
        args.output = f"{input_name}_upscaled"
    
    print(f"Input: {args.input}")
    print(f"Semantic file: {args.semantic}")
    print(f"Material mapping: {args.material_mapping}")
    print(f"Generated textures dir: {args.generated_textures_dir}")
    print(f"Output: {args.output}")
    print(f"Upscale factor: {args.upscale_factor}")
    print(f"Source texture scale: {args.source_texture_scale}")
    
    upscaler = TextureUpscaler(
        semantic_file=args.semantic,
        material_mapping_file=args.material_mapping,
        generated_textures_dir=args.generated_textures_dir,
        upscale_factor=args.upscale_factor, 
        texture_size=args.texture_size,
        source_texture_scale=args.source_texture_scale
    )
    upscaler.upscale_texture(args.input, args.output)

if __name__ == "__main__":
    main()