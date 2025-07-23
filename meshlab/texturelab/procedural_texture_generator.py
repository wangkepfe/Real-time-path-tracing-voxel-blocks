#!/usr/bin/env python3
"""
Procedural PBR Texture Generator

Generates seamless/tilable PBR textures (albedo, normal, roughness)
for various materials at 512x512 resolution representing 100 sq cm real world scale.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import os
import math
import random

class ProceduralTextureGenerator:
    def __init__(self, size=512):
        self.size = size
        self.output_dir = "generated_textures"
        os.makedirs(self.output_dir, exist_ok=True)

    def tileable_value_noise(self, size, grid_size):
        """Generate tileable value noise"""
        width, height = size
        gx, gy = grid_size
        # Random grid - note the order of dimensions
        grid = np.random.rand(gy, gx)  # Changed order to match the expected numpy convention

        # Smoothstep function
        def smoothstep(t):
            return 3 * t**2 - 2 * t**3

        # Coordinate arrays
        xi = (np.arange(width) / width) * gx
        yi = (np.arange(height) / height) * gy
        x = xi[:, None]
        y = yi[None, :]

        # Integer and fractional parts
        x0 = np.floor(x).astype(int) % gx
        y0 = np.floor(y).astype(int) % gy
        x1 = (x0 + 1) % gx
        y1 = (y0 + 1) % gy
        tx = smoothstep(x - x0)
        ty = smoothstep(y - y0)

        # Sample grid corners
        n00 = grid[y0, x0]
        n10 = grid[y1, x0]
        n01 = grid[y0, x1]
        n11 = grid[y1, x1]

        # Interpolate
        nx0 = n00 * (1 - ty) + n10 * ty
        nx1 = n01 * (1 - ty) + n11 * ty
        return nx0 * (1 - tx) + nx1 * tx

    def generate_height_map(self, size, octaves=4, base_grid=4):
        """Generate fractal noise height map"""
        width, height = size
        height_map = np.zeros((width, height), dtype=np.float32)
        amplitude = 1.0
        frequency = 1
        total_amp = 0.0

        for _ in range(octaves):
            # Handle both single integer and tuple grid sizes
            if isinstance(base_grid, tuple):
                grid_size = (int(base_grid[0]*frequency), int(base_grid[1]*frequency))
            else:
                grid_size = (int(base_grid*frequency), int(base_grid*frequency))
            noise = self.tileable_value_noise(size, grid_size)
            height_map += noise * amplitude
            total_amp += amplitude
            amplitude *= 0.5
            frequency *= 2

        height_map /= total_amp
        return height_map

    def generate_noise(self, octaves=4, persistence=0.5, scale=50.0, seamless=True):
        """Generate seamless Perlin-like noise"""
        noise = np.zeros((self.size, self.size))

        for i in range(octaves):
            freq = 2 ** i / scale
            amp = persistence ** i

            # Create coordinate grids
            x = np.linspace(0, freq * 2 * np.pi, self.size, endpoint=seamless)
            y = np.linspace(0, freq * 2 * np.pi, self.size, endpoint=seamless)
            X, Y = np.meshgrid(x, y)

            # Generate noise using trigonometric functions for seamless tiling
            noise_layer = np.sin(X) * np.cos(Y) + np.sin(X + np.pi/3) * np.cos(Y + np.pi/3)
            noise_layer += np.sin(X * 1.3) * np.cos(Y * 0.7) * 0.5

            noise += amp * noise_layer

        return (noise - noise.min()) / (noise.max() - noise.min())

    def generate_fbm_noise(self, octaves=6, persistence=0.5, scale=30.0):
        """Generate fractal Brownian motion noise"""
        noise = np.zeros((self.size, self.size))
        amplitude = 1.0
        frequency = scale
        max_value = 0.0

        for i in range(octaves):
            x = np.linspace(0, frequency * 2 * np.pi, self.size, endpoint=False)
            y = np.linspace(0, frequency * 2 * np.pi, self.size, endpoint=False)
            X, Y = np.meshgrid(x, y)

            # Multi-layer noise
            layer = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2*X) * np.cos(2*Y)
            layer += 0.25 * np.sin(4*X + np.pi/4) * np.cos(4*Y + np.pi/4)

            noise += amplitude * layer
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2

        return noise / max_value

    def create_normal_from_height(self, height_map, strength=1.0):
        """Create normal map from height map"""
        height_map = np.array(height_map, dtype=np.float32)

        # Calculate gradients
        grad_x = np.gradient(height_map, axis=1) * strength
        grad_y = np.gradient(height_map, axis=0) * strength

        # Create normal vectors
        normal_x = -grad_x
        normal_y = -grad_y
        normal_z = np.ones_like(grad_x)

        # Normalize
        length = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
        normal_x /= length
        normal_y /= length
        normal_z /= length

        # Convert to 0-255 range
        normal_map = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        normal_map[:, :, 0] = ((normal_x + 1) * 127.5).astype(np.uint8)  # Red = X
        normal_map[:, :, 1] = ((normal_y + 1) * 127.5).astype(np.uint8)  # Green = Y
        normal_map[:, :, 2] = ((normal_z + 1) * 127.5).astype(np.uint8)  # Blue = Z

        return normal_map

    def save_texture_set(self, name, albedo, normal, roughness):
        """Save a complete PBR texture set"""
        albedo_img = Image.fromarray(albedo.astype(np.uint8))
        normal_img = Image.fromarray(normal.astype(np.uint8))
        roughness_img = Image.fromarray(roughness.astype(np.uint8))

        albedo_img.save(f"{self.output_dir}/{name}_albedo.png")
        normal_img.save(f"{self.output_dir}/{name}_normal.png")
        roughness_img.save(f"{self.output_dir}/{name}_roughness.png")

        print(f"Generated {name} texture set")

    def generate_human_skin(self):
        """Generate human skin texture"""
        size = (self.size, self.size)

        # Create base height map for skin texture
        height = self.generate_height_map(size, octaves=5, base_grid=6)

        # Add finer pore detail
        pore_height = self.generate_height_map(size, octaves=3, base_grid=12)
        height = height * 0.7 + pore_height * 0.3

        # Compute normal map from height map
        gy, gx = np.gradient(height)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        norm = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal / norm + 1) / 2  # Normalize and remap to 0-1
        normal = (normal * 255).astype(np.uint8)

        # Create albedo: base skin tone with subtle variation from height
        base_tone = np.array([220, 180, 140], dtype=np.float32)  # Peachy skin tone
        variation = (height[:, :, None] - 0.5) * 5  # Reduced from 30 to 15 for less contrast
        # Add slight redness in pore areas
        redness = (pore_height[:, :, None] - 0.5) * 10  # Reduced from 20 to 10
        red_variation = np.zeros_like(variation)
        red_variation[:, :, 0] = redness[:, :, 0]  # Add to red channel

        albedo = np.clip(base_tone + variation + red_variation, 0, 255).astype(np.uint8)

        # Create roughness: combine base height with pore detail
        # Reduced variation range and adjusted base value
        roughness = np.clip(0.45 + (height - 0.5) * 0.025 + (pore_height - 0.5) * 0.05, 0, 1)  # Reduced multipliers
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("human_skin", albedo, normal, roughness)

    def generate_hair(self):
        """Generate simplified hair texture with vertical strands"""
        size = (self.size, self.size)

        # Generate directional fractal noise (horizontal strand effect)
        def generate_hair_height(size, octaves=4, base_grid=(1, 16)):  # Flipped grid dimensions
            height_map = np.zeros(size, dtype=np.float32)
            amplitude = 1.0
            total_amp = 0.0
            freq = 1
            for _ in range(octaves):
                grid = (int(base_grid[0]*freq), int(base_grid[1]*freq))
                height_map += self.tileable_value_noise(size, grid) * amplitude
                total_amp += amplitude
                amplitude *= 0.5
                freq *= 2
            return height_map / total_amp

        # Generate base height map
        height = generate_hair_height(size)

        # Add some cross-strand variation
        cross_height = generate_hair_height(size, base_grid=(8, 2))  # Some cross-strand detail
        height = height * 0.8 + cross_height * 0.2

        # Normal map
        gy, gx = np.gradient(height)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        n = np.linalg.norm(normal, axis=2, keepdims=True)
        normal = ((normal / n) + 1) / 2
        normal = (normal * 255).astype(np.uint8)

        # Albedo map (brown base with strand variation)
        base_tone = np.array([80, 50, 30], dtype=np.float32)  # Dark brown base
        variation = (height - 0.5)[..., None] * 20
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Roughness map (low gloss with subtle variation)
        roughness = np.clip(0.2 + (height - 0.5) * 0.05, 0, 1)
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("hair", albedo, normal, roughness)

    def generate_shiny_silver_metal(self):
        """Generate shiny silver metal texture (constant values)"""
        # Create uniform arrays for each texture map

        # Albedo - polished silver has a slight blue-white tint
        # Using standard silver color values (slightly cool-tinted white)
        albedo = np.full((self.size, self.size, 3), [192, 192, 192], dtype=np.uint8)

        # Normal map - flat surface (pointing straight up)
        normal = np.full((self.size, self.size, 3), [127, 127, 255], dtype=np.uint8)

        # Roughness - very low for polished silver (highly reflective)
        # Value around 0.1 for a polished finish
        roughness = np.full((self.size, self.size), int(0.1 * 255), dtype=np.uint8)

        self.save_texture_set("shiny_silver_metal", albedo, normal, roughness)

    def generate_leather(self):
        """Generate leather texture"""
        size = (self.size, self.size)

        # Macro wrinkles (low-frequency)
        macro = self.generate_height_map(size, octaves=4, base_grid=4)

        # Micro pores (high-frequency)
        micro = self.generate_height_map(size, octaves=4, base_grid=64)

        # Combined height map
        height_map = np.clip(0.5 + (macro - 0.5) * 0.4 + (micro - 0.5) * 0.1, 0, 1)

        # Compute normal map
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Albedo: warm leather base with macro shading
        base_tone = np.array([120, 80, 60], dtype=np.float32)  # Warm brown leather
        variation = (macro - 0.5)[..., None] * 30 + (micro - 0.5)[..., None] * 10
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Roughness: medium-high with pore and wrinkle modulation
        rough_base = 0.6  # Medium-high roughness for regular leather
        rough_map = np.clip(rough_base + (macro - 0.5) * 0.2 - (micro - 0.5) * 0.1, 0, 1)
        roughness = (rough_map * 255).astype(np.uint8)

        self.save_texture_set("leather", albedo, normal, roughness)

    def generate_ribbon(self):
        """Generate ribbon texture with directional ridges and micro detail"""
        width, height = self.size, self.size
        size = (height, width)

        # Generate anisotropic ridges (horizontal lines)
        ridges = self.generate_height_map(size, octaves=4, base_grid=(64, 1))

        # Generate random micro bumps for fabric look
        micro = self.generate_height_map(size, octaves=4, base_grid=(64, 64))

        # Construct height map: directional ridges + subtle micro texture
        height_map = np.clip(
            0.6 * (ridges - 0.5) * 0.2 +  # directional ridge height variation
            0.3 * (micro - 0.5) * 0.1 + 0.5, 0, 1)

        # Compute normal map from height
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Create albedo: soft pink ribbon with faint directional shading
        base_tone = np.array([255, 240, 245], dtype=np.float32)  # Light pink
        variation = np.zeros((height, width, 3), dtype=np.float32)
        # Apply variation to all channels
        for i in range(3):
            variation[:, :, i] = (ridges - 0.5) * 15 + (micro - 0.5) * 5
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Create roughness: low roughness for slight sheen, with directional variation
        rough_base = 0.2  # Low roughness for silky ribbon
        roughness = np.clip(
            rough_base +
            (ridges - 0.5) * -0.05 +  # ridges are shinier
            (micro - 0.5) * 0.02, 0, 1)  # subtle micro variation
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("ribbon", albedo, normal, roughness)

    def generate_shiny_black_leather(self):
        """Generate shiny black leather texture"""
        size = (self.size, self.size)

        # Macro wrinkles (low-frequency, less pronounced)
        macro = self.generate_height_map(size, octaves=4, base_grid=4)

        # Micro pores (high-frequency, subtle)
        micro = self.generate_height_map(size, octaves=4, base_grid=64)

        # Combined height map (reduced strength for smoother appearance)
        height_map = np.clip(0.5 + (macro - 0.5) * 0.2 + (micro - 0.5) * 0.05, 0, 1)

        # Compute normal map
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Albedo: black leather base with subtle variation
        base_tone = np.array([25, 20, 20], dtype=np.float32)  # Deep black
        variation = (macro - 0.5)[..., None] * 15 + (micro - 0.5)[..., None] * 5  # Subtle variation
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Roughness: low base with subtle variation for shiny appearance
        rough_base = 0.2  # Low roughness for shiny leather
        rough_map = np.clip(rough_base + (macro - 0.5) * 0.1 - (micro - 0.5) * 0.05, 0, 1)
        roughness = (rough_map * 255).astype(np.uint8)

        self.save_texture_set("shiny_black_leather", albedo, normal, roughness)

    def generate_pink_terry_cloth(self):
        """Generate pink cotton terry cloth fabric with wavy ribs and fiber detail"""
        width, height = self.size, self.size
        size = (width, height)

        # Create normalized coordinate grids
        x = np.linspace(0, 1, width, endpoint=False)
        y = np.linspace(0, 1, height, endpoint=False)
        X, Y = np.meshgrid(x, y)

        # Generate micro-scale fiber fuzz
        micro = self.generate_height_map(size, octaves=4, base_grid=64)

        # Generate warp noise for non-uniform wavy ribs
        warp_noise = self.generate_height_map(size, octaves=3, base_grid=(1, 16))

        # Create macro wavy vertical ribs
        N = 16  # number of ribs
        warp_amp = 0.02  # amplitude of the warp
        macro = np.sin(2 * np.pi * N * (X + (warp_noise - 0.5) * warp_amp))
        macro_norm = (macro + 1) * 0.5

        # Combined height map: ribs + fuzz
        height_map = np.clip(0.1 + 0.6 * macro_norm + 0.3 * micro, 0, 1)

        # Compute normal map from height map
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Create albedo: pink base with subtle shading
        base_tone = np.array([255, 180, 200], dtype=np.float32)  # Pink base
        variation = np.zeros((height, width, 3), dtype=np.float32)
        # Apply variation to all channels
        for i in range(3):
            variation[:, :, i] = (macro_norm - 0.5) * 10 + (micro - 0.5) * 10
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Create roughness: matte with subtle variation
        rough_base = 0.7  # Base roughness for terry cloth
        roughness = np.clip(rough_base +
                          (micro - 0.5) * 0.1 -  # Micro fiber variation
                          (macro_norm - 0.5) * 0.05,  # Rib variation
                          0, 1)
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("pink_terry_cloth", albedo, normal, roughness)

    def generate_cotton_plaid(self):
        """Generate natural cotton plaid fabric texture"""
        width, height = self.size, self.size
        size = (width, height)

        # Create grid coordinates
        x = np.linspace(0, 1, width, endpoint=False)
        y = np.linspace(0, 1, height, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='xy')

        # Plaid parameters
        nx, ny = 8, 8  # Number of stripes
        wx, wy = 0.2, 0.15  # Stripe widths

        # Generate stripe masks
        mask_x = ((X * nx) % 1) < wx
        mask_y = ((Y * ny) % 1) < wy
        mask_xy = mask_x & mask_y

        # Gray levels for plaid pattern
        base_gray = 200  # light gray
        stripe_gray = 80  # darker gray
        overlap_gray = 40  # darkest gray

        # Create albedo with plaid pattern
        albedo_gray = np.full((height, width), base_gray, dtype=np.uint8)
        albedo_gray[mask_x] = stripe_gray
        albedo_gray[mask_y] = stripe_gray
        albedo_gray[mask_xy] = overlap_gray
        albedo = np.dstack([albedo_gray] * 3)

        # Generate micro-detail for fabric texture
        micro = self.tileable_value_noise(size, (256, 256))

        # Height map combining plaid pattern and micro detail
        height_map = np.clip(0.05 * micro +
                           0.02 * mask_x.astype(float) +
                           0.02 * mask_y.astype(float), 0, 1)

        # Compute normal map
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Create roughness map
        rough_base = 0.8  # Base roughness for cotton
        roughness = np.clip(rough_base - micro * 0.1, 0, 1)  # Subtle variation from weave
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("cotton_plaid", albedo, normal, roughness)

    def generate_white_cotton(self):
        """Generate white cotton fabric texture"""
        size = (self.size, self.size)

        # Generate micro-scale pore detail
        micro = self.generate_height_map(size, octaves=4, base_grid=64)

        # Height map for normals (slight amplitude)
        height_map = np.clip((micro - 0.5) * 0.1 + 0.5, 0, 1)

        # Compute normal map
        gy, gx = np.gradient(height_map)
        normal = np.dstack((gx, gy, np.ones_like(gx)))
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        normal = (normal + 1) * 0.5
        normal = (normal * 255).astype(np.uint8)

        # Albedo: plain cotton base with subtle micro shading
        base_tone = np.array([230, 230, 230], dtype=np.float32)  # Bright white cotton
        variation = (micro - 0.5)[..., None] * 20  # Subtle variation
        albedo = np.clip(base_tone + variation, 0, 255).astype(np.uint8)

        # Roughness: matte cotton, roughness ~0.9 ± variation
        rough_base = 0.9  # High roughness for matte cotton
        roughness = np.clip(rough_base - (micro - 0.5) * 0.1, 0, 1)  # Subtle variation
        roughness = (roughness * 255).astype(np.uint8)

        self.save_texture_set("white_cotton", albedo, normal, roughness)

    def generate_all_textures(self):
        """Generate all texture sets"""
        print("Generating procedural PBR textures...")
        print(f"Resolution: {self.size}x{self.size} (representing 100 sq cm)")
        print(f"Output directory: {self.output_dir}")
        print()

        self.generate_human_skin()
        self.generate_hair()
        self.generate_shiny_silver_metal()
        self.generate_leather()
        self.generate_ribbon()
        self.generate_shiny_black_leather()
        self.generate_pink_terry_cloth()
        self.generate_cotton_plaid()
        self.generate_white_cotton()

        print("\nAll textures generated successfully!")
        print(f"Total: 27 texture files (9 materials × 3 maps each)")

if __name__ == "__main__":
    generator = ProceduralTextureGenerator(size=512)
    generator.generate_all_textures()