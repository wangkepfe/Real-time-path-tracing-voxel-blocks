#!/usr/bin/env python3
"""
Minecraft Character GLTF Generator

Generates GLTF files for Minecraft characters based on the specification
in minecraft_character.json and a given texture.
"""

import json
import struct
import base64
import numpy as np
from PIL import Image
import os
import sys
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Face:
    """Represents a face of a cube with texture coordinates"""
    vertices: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    indices: List[int]

@dataclass
class Mesh:
    """Represents a mesh with vertices, UVs, indices, and skinning data"""
    vertices: List[Tuple[float, float, float]]
    uvs: List[Tuple[float, float]]
    indices: List[int]
    joint_indices: List[Tuple[int, int, int, int]]  # 4 joint indices per vertex
    joint_weights: List[Tuple[float, float, float, float]]  # 4 weights per vertex

class MinecraftCharacterGenerator:
    def __init__(self, spec_file: str, scale_factor: float = 2.0 / 32.0):
        """Initialize the generator with specification file and optional scale factor"""
        with open(spec_file, 'r') as f:
            self.spec = json.load(f)

        self.texture_size = self.spec.get('textureSize', {'width': 64, 'height': 64})
        self.parts = self.spec['parts']
        self.skeleton = self.spec.get('skeleton', [])
        self.scale_factor = scale_factor
        
        # Load animations from external file or inline
        self.animations = self._load_animations()

        # Output mesh data
        self.vertices = []
        self.uvs = []
        self.indices = []
        self.joint_indices = []  # 4 joint indices per vertex
        self.joint_weights = []  # 4 weights per vertex
        self.current_vertex_index = 0

        # Create bone name to index mapping
        self.bone_to_index = {}
        for i, bone in enumerate(self.skeleton):
            self.bone_to_index[bone['name']] = i

    def _load_animations(self) -> Dict:
        """Load animations from external file or inline specification"""
        # Check if animations are specified inline
        if 'animations' in self.spec:
            return self.spec['animations']
        
        # Check if there's an animationsFile reference
        if 'animationsFile' in self.spec:
            animations_file = self.spec['animationsFile']
            try:
                with open(animations_file, 'r') as f:
                    animations_data = json.load(f)
                return animations_data.get('animations', {})
            except FileNotFoundError:
                print(f"Warning: Animation file {animations_file} not found, using no animations")
                return {}
        
        # Try to load default animations file
        try:
            with open('minecraft_character_animations.json', 'r') as f:
                animations_data = json.load(f)
            return animations_data.get('animations', {})
        except FileNotFoundError:
            print("Warning: No animations file found, using no animations")
            return {}

    def normalize_uv(self, u: int, v: int) -> Tuple[float, float]:
        """Convert texture pixel coordinates to normalized UV coordinates"""
        return (u / self.texture_size['width'], v / self.texture_size['height'])

    def scale_vertices_from_center(self, vertices: List[Tuple[float, float, float]], aabb: Dict, scale_factor: float) -> List[Tuple[float, float, float]]:
        """Scale vertices from the center of the AABB"""
        min_x, min_y, min_z = aabb['min']
        max_x, max_y, max_z = aabb['max']

        # Calculate center of the AABB
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        center_z = (min_z + max_z) / 2.0

        scaled_vertices = []
        for x, y, z in vertices:
            # Translate to origin, scale, then translate back
            scaled_x = center_x + (x - center_x) * scale_factor
            scaled_y = center_y + (y - center_y) * scale_factor
            scaled_z = center_z + (z - center_z) * scale_factor
            scaled_vertices.append((scaled_x, scaled_y, scaled_z))

        return scaled_vertices

    def create_cube_face(self, aabb: Dict, face_name: str, texture_coords: List[int]) -> Face:
        """Create a single face of a cube with proper texture coordinates"""
        min_x, min_y, min_z = aabb['min']
        max_x, max_y, max_z = aabb['max']

        # Define face vertices based on face direction
        if face_name == 'front':  # +Z face
            vertices = [
                (min_x, min_y, max_z),
                (max_x, min_y, max_z),
                (max_x, max_y, max_z),
                (min_x, max_y, max_z)
            ]
        elif face_name == 'back':  # -Z face
            vertices = [
                (max_x, min_y, min_z),
                (min_x, min_y, min_z),
                (min_x, max_y, min_z),
                (max_x, max_y, min_z)
            ]
        elif face_name == 'left':
            vertices = [
                (max_x, min_y, max_z),
                (max_x, min_y, min_z),
                (max_x, max_y, min_z),
                (max_x, max_y, max_z)
            ]
        elif face_name == 'right':
            vertices = [
                (min_x, min_y, min_z),
                (min_x, min_y, max_z),
                (min_x, max_y, max_z),
                (min_x, max_y, min_z)
            ]
        elif face_name == 'top':  # +Y face
            vertices = [
                (min_x, max_y, max_z),
                (max_x, max_y, max_z),
                (max_x, max_y, min_z),
                (min_x, max_y, min_z)
            ]
        elif face_name == 'bottom':  # -Y face
            vertices = [
                (min_x, min_y, min_z),
                (max_x, min_y, min_z),
                (max_x, min_y, max_z),
                (min_x, min_y, max_z)
            ]

        # Convert texture coordinates to UV
        u1, v1, u2, v2 = texture_coords
        uvs = [
            self.normalize_uv(u1, v2),  # bottom-left
            self.normalize_uv(u2, v2),  # bottom-right
            self.normalize_uv(u2, v1),  # top-right
            self.normalize_uv(u1, v1)   # top-left
        ]

        if face_name == 'bottom':
            uvs = [
                self.normalize_uv(u1, v1),  # top-left
                self.normalize_uv(u2, v1),  # top-right
                self.normalize_uv(u2, v2),  # bottom-right
                self.normalize_uv(u1, v2)   # bottom-left
            ]

        # Apply scale factor to vertices
        scaled_vertices = [(x * self.scale_factor, y * self.scale_factor, z * self.scale_factor)
                          for x, y, z in vertices]

        # Create triangle indices for the quad (0,1,2) and (0,2,3)
        indices = [0, 1, 2, 0, 2, 3]

        return Face(scaled_vertices, uvs, indices)

    def add_face_to_mesh(self, face: Face, bone_name: str = None):
        """Add a face to the current mesh with bone assignment"""
        base_index = self.current_vertex_index

        # Add vertices and UVs
        self.vertices.extend(face.vertices)
        self.uvs.extend(face.uvs)

        # Add joint indices and weights for each vertex
        joint_index = self.bone_to_index.get(bone_name, 0) if bone_name else 0
        for _ in face.vertices:
            # Each vertex is influenced by one bone with full weight
            self.joint_indices.append((joint_index, 0, 0, 0))
            self.joint_weights.append((1.0, 0.0, 0.0, 0.0))

        # Add indices with offset
        for idx in face.indices:
            self.indices.append(base_index + idx)

        self.current_vertex_index += len(face.vertices)

    def create_base_layer_part(self, part: Dict):
        """Create geometry for a base layer part (simple cube)"""
        aabb = part['aabb'].copy()  # Make a copy to avoid modifying original

        # Shrink AABB by 0.005 units from all sides to create 0.01 gap between parts
        gap_size = 0.005
        aabb['min'] = [aabb['min'][0] + gap_size, aabb['min'][1] + gap_size, aabb['min'][2] + gap_size]
        aabb['max'] = [aabb['max'][0] - gap_size, aabb['max'][1] - gap_size, aabb['max'][2] - gap_size]

        faces = part['faces']
        bone_name = part.get('bone')

        for face_name, texture_coords in faces.items():
            face = self.create_cube_face(aabb, face_name, texture_coords)
            self.add_face_to_mesh(face, bone_name)

    def get_texture_alpha(self, texture: Image.Image, u: int, v: int) -> int:
        """Get alpha value at texture coordinate (u, v)"""
        if u >= texture.width or v >= texture.height or u < 0 or v < 0:
            return 0

        pixel = texture.getpixel((u, v))
        if isinstance(pixel, tuple) and len(pixel) >= 4:
            return pixel[3]  # Alpha channel
        elif isinstance(pixel, tuple) and len(pixel) == 3:
            return 255  # No alpha channel, assume opaque
        else:
            return 255  # Single channel, assume opaque

    def create_overlay_quad(self, aabb: Dict, face_name: str, u: int, v: int, texture_coords: List[int]) -> Face:
        """Create a single 1x1 unit quad for one texel in overlay geometry"""
        min_x, min_y, min_z = aabb['min']
        max_x, max_y, max_z = aabb['max']

        # Get texture coordinates for this face (not normalized)
        tex_u1, tex_v1, tex_u2, tex_v2 = texture_coords

        # Calculate texel position relative to texture coordinates
        texel_u = u - tex_u1
        texel_v = v - tex_v1

        # Create a 1x1 unit quad positioned on the face surface
        # Each texel maps to 1 unit in world space
        if face_name == 'front':  # +Z face
            local_x1 = min_x + texel_u
            local_x2 = local_x1 + 1
            local_y1 = max_y - (texel_v + 1)
            local_y2 = local_y1 + 1

            vertices = [
                (local_x1, local_y1, max_z),
                (local_x2, local_y1, max_z),
                (local_x2, local_y2, max_z),
                (local_x1, local_y2, max_z)
            ]

        elif face_name == 'back':  # -Z face
            local_x1 = max_x - (texel_u + 1)
            local_x2 = local_x1 + 1
            local_y1 = max_y - (texel_v + 1)
            local_y2 = local_y1 + 1

            vertices = [
                (local_x2, local_y1, min_z),
                (local_x1, local_y1, min_z),
                (local_x1, local_y2, min_z),
                (local_x2, local_y2, min_z)
            ]

        elif face_name == 'left':  # +X face
            local_z1 = max_z - (texel_u + 1)
            local_z2 = local_z1 + 1
            local_y1 = max_y - (texel_v + 1)
            local_y2 = local_y1 + 1

            vertices = [
                (max_x, local_y1, local_z2),
                (max_x, local_y1, local_z1),
                (max_x, local_y2, local_z1),
                (max_x, local_y2, local_z2)
            ]

        elif face_name == 'right':  # -X face
            local_z1 = min_z + texel_u
            local_z2 = local_z1 + 1
            local_y1 = max_y - (texel_v + 1)
            local_y2 = local_y1 + 1

            vertices = [
                (min_x, local_y1, local_z1),
                (min_x, local_y1, local_z2),
                (min_x, local_y2, local_z2),
                (min_x, local_y2, local_z1)
            ]

        elif face_name == 'top':  # +Y face
            local_x1 = min_x + texel_u
            local_x2 = local_x1 + 1
            local_z1 = min_z + texel_v
            local_z2 = local_z1 + 1

            vertices = [
                (local_x1, max_y, local_z1),
                (local_x1, max_y, local_z2),
                (local_x2, max_y, local_z2),
                (local_x2, max_y, local_z1)
            ]

        elif face_name == 'bottom':  # -Y face
            local_x1 = min_x + texel_u
            local_x2 = local_x1 + 1
            local_z1 = min_z + texel_v
            local_z2 = local_z1 + 1

            vertices = [
                (local_x1, min_y, local_z2),
                (local_x1, min_y, local_z1),
                (local_x2, min_y, local_z1),
                (local_x2, min_y, local_z2)
            ]

        # Scale overlay vertices by 1.1x from center of the part
        vertices = self.scale_vertices_from_center(vertices, aabb, 1.1)

        # Apply global scale factor to vertices
        scaled_vertices = [(x * self.scale_factor, y * self.scale_factor, z * self.scale_factor)
                          for x, y, z in vertices]

        # UV coordinates for this single texel (normalized for final output)
        u_norm, v_norm = self.normalize_uv(u, v)
        u_next, v_next = self.normalize_uv(u + 1, v + 1)

        uvs = [
            (u_norm, v_next),
            (u_next, v_next),
            (u_next, v_norm),
            (u_norm, v_norm)
        ]

        indices = [0, 1, 2, 0, 2, 3]
        return Face(scaled_vertices, uvs, indices)

    def create_overlay_layer_part(self, part: Dict, texture: Image.Image):
        """Create geometry for an overlay layer part (per-texel quads)"""
        aabb = part['aabb']
        faces = part['faces']

        # Find corresponding base part with same name for bone assignment
        bone_name = part.get('bone')
        if not bone_name:
            # Look for base layer part with same name
            for base_part in self.parts:
                if (base_part['name'] == part['name'] and
                    base_part['layer'] == 'base' and
                    'bone' in base_part):
                    bone_name = base_part['bone']
                    break

        for face_name, texture_coords in faces.items():
            u1, v1, u2, v2 = texture_coords

            # Check each texel in the face region
            for v in range(v1, v2):
                for u in range(u1, u2):
                    alpha = self.get_texture_alpha(texture, u, v)
                    if alpha > 0:  # Create quad only if texel is not transparent
                        quad = self.create_overlay_quad(aabb, face_name, u, v, texture_coords)
                        self.add_face_to_mesh(quad, bone_name)

    def generate_mesh(self, texture_path: str) -> Mesh:
        """Generate the complete mesh for the character"""
        # Load texture
        try:
            texture = Image.open(texture_path).convert('RGBA')
        except Exception as e:
            print(f"Warning: Could not load texture {texture_path}: {e}")
            # Create a dummy texture
            texture = Image.new('RGBA', (64, 64), (255, 255, 255, 255))

        # Reset mesh data
        self.vertices = []
        self.uvs = []
        self.indices = []
        self.joint_indices = []
        self.joint_weights = []
        self.current_vertex_index = 0

        # Process all parts
        for part in self.parts:
            if part['layer'] == 'base':
                self.create_base_layer_part(part)
            if part['layer'] == 'overlay':
                self.create_overlay_layer_part(part, texture)

        return Mesh(self.vertices, self.uvs, self.indices, self.joint_indices, self.joint_weights)

    def create_all_animation_data(self) -> Dict:
        """Create animation data for all animations from JSON specification"""
        if not self.animations:
            # Return empty animation data if no animations defined
            return {
                'animations': [],
                'has_animations': False
            }

        animations_data = []
        
        for anim_name, animation in self.animations.items():
            keyframes = animation['keyframes']
            bones = animation['bones']
            interpolation = animation.get('interpolation', 'LINEAR')

            # Convert keyframes to binary data
            time_data = np.array(keyframes, dtype=np.float32).tobytes()

            # Extract bone transformation data
            hip_translations = bones.get('hip', {}).get('translation', [[0, 0, 0]] * len(keyframes))
            torso_rotations = bones.get('torso', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))
            head_rotations = bones.get('head', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))
            right_leg_rotations = bones.get('right_leg', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))
            left_leg_rotations = bones.get('left_leg', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))
            right_arm_rotations = bones.get('right_arm', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))
            left_arm_rotations = bones.get('left_arm', {}).get('rotation', [[0, 0, 0, 1]] * len(keyframes))

            # Apply scale factor to translation data
            scaled_hip_translations = [[t[0] * self.scale_factor, t[1] * self.scale_factor, t[2] * self.scale_factor]
                                      for t in hip_translations]

            # Handle CUBICSPLINE vs LINEAR data format
            if interpolation == 'CUBICSPLINE':
                # For CUBICSPLINE, we need in-tangent, value, out-tangent for each keyframe
                # Since our JSON only has values, we'll create zero tangents for smooth interpolation
                
                # Expand hip translations: [in_tangent, value, out_tangent] for each keyframe
                expanded_hip_translations = []
                for translation in scaled_hip_translations:
                    expanded_hip_translations.extend([[0, 0, 0], translation, [0, 0, 0]])
                
                # Expand rotations: [in_tangent, value, out_tangent] for each keyframe
                expanded_torso_rotations = []
                for rotation in torso_rotations:
                    expanded_torso_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                expanded_head_rotations = []
                for rotation in head_rotations:
                    expanded_head_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                expanded_right_leg_rotations = []
                for rotation in right_leg_rotations:
                    expanded_right_leg_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                expanded_left_leg_rotations = []
                for rotation in left_leg_rotations:
                    expanded_left_leg_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                expanded_right_arm_rotations = []
                for rotation in right_arm_rotations:
                    expanded_right_arm_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                expanded_left_arm_rotations = []
                for rotation in left_arm_rotations:
                    expanded_left_arm_rotations.extend([[0, 0, 0, 0], rotation, [0, 0, 0, 0]])
                
                # Convert to binary data
                hip_translation_data = np.array(expanded_hip_translations, dtype=np.float32).tobytes()
                torso_rotation_data = np.array(expanded_torso_rotations, dtype=np.float32).tobytes()
                head_rotation_data = np.array(expanded_head_rotations, dtype=np.float32).tobytes()
                right_leg_rotation_data = np.array(expanded_right_leg_rotations, dtype=np.float32).tobytes()
                left_leg_rotation_data = np.array(expanded_left_leg_rotations, dtype=np.float32).tobytes()
                right_arm_rotation_data = np.array(expanded_right_arm_rotations, dtype=np.float32).tobytes()
                left_arm_rotation_data = np.array(expanded_left_arm_rotations, dtype=np.float32).tobytes()
                
                # Update keyframe count for accessors (3x original for CUBICSPLINE)
                accessor_keyframe_count = len(keyframes) * 3
            else:
                # LINEAR interpolation - use data as-is
                hip_translation_data = np.array(scaled_hip_translations, dtype=np.float32).tobytes()
                torso_rotation_data = np.array(torso_rotations, dtype=np.float32).tobytes()
                head_rotation_data = np.array(head_rotations, dtype=np.float32).tobytes()
                right_leg_rotation_data = np.array(right_leg_rotations, dtype=np.float32).tobytes()
                left_leg_rotation_data = np.array(left_leg_rotations, dtype=np.float32).tobytes()
                right_arm_rotation_data = np.array(right_arm_rotations, dtype=np.float32).tobytes()
                left_arm_rotation_data = np.array(left_arm_rotations, dtype=np.float32).tobytes()
                
                # Use original keyframe count
                accessor_keyframe_count = len(keyframes)

            animation_data = {
                'name': anim_name,
                'display_name': animation.get('name', anim_name),
                'duration': animation.get('duration', 1.0),
                'interpolation': interpolation,
                'keyframe_count': len(keyframes),
                'accessor_keyframe_count': accessor_keyframe_count,
                'time_data': time_data,
                'hip_translation_data': hip_translation_data,
                'torso_rotation_data': torso_rotation_data,
                'head_rotation_data': head_rotation_data,
                'right_leg_rotation_data': right_leg_rotation_data,
                'left_leg_rotation_data': left_leg_rotation_data,
                'right_arm_rotation_data': right_arm_rotation_data,
                'left_arm_rotation_data': left_arm_rotation_data
            }
            
            animations_data.append(animation_data)

        return {
            'animations': animations_data,
            'has_animations': len(animations_data) > 0
        }

    def create_inverse_bind_matrices(self) -> np.ndarray:
        """Create inverse bind matrices for each joint"""
        num_joints = len(self.skeleton)
        inverse_bind_matrices = np.zeros((num_joints, 16), dtype=np.float32)

        for i, bone in enumerate(self.skeleton):
            # The JSON already provides world space positions in 'head'
            bone_world_pos = bone['head']

            # Apply global scaling
            scaled_pos = [pos * self.scale_factor for pos in bone_world_pos]

            # Create inverse bind matrix (inverse of bone's bind pose transform)
            # In column-major order (OpenGL/GLTF convention):
            # Elements 0-3: first column, 4-7: second column, etc.
            # Translation is in elements 12, 13, 14
            matrix = np.eye(4, dtype=np.float32)
            matrix[0, 3] = -scaled_pos[0]  # Translation X in column-major
            matrix[1, 3] = -scaled_pos[1]  # Translation Y in column-major
            matrix[2, 3] = -scaled_pos[2]  # Translation Z in column-major

            # Flatten to column-major order for GLTF (transpose before flattening)
            inverse_bind_matrices[i] = matrix.T.flatten()

        return inverse_bind_matrices

    def mesh_to_gltf(self, mesh: Mesh, output_path: str, texture_path: str):
        """Export mesh to GLTF format"""
        # Convert mesh data to numpy arrays for binary packing
        vertices = np.array(mesh.vertices, dtype=np.float32)
        uvs = np.array(mesh.uvs, dtype=np.float32)
        indices = np.array(mesh.indices, dtype=np.uint16)
        joint_indices = np.array(mesh.joint_indices, dtype=np.uint16)
        joint_weights = np.array(mesh.joint_weights, dtype=np.float32)

        # Create animation data for all animations
        all_animations = self.create_all_animation_data()

        # Create inverse bind matrices
        inverse_bind_matrices = self.create_inverse_bind_matrices()

        # Pack binary data
        vertex_data = vertices.tobytes()
        uv_data = uvs.tobytes()
        index_data = indices.tobytes()
        joint_indices_data = joint_indices.tobytes()
        joint_weights_data = joint_weights.tobytes()
        inverse_bind_matrices_data = inverse_bind_matrices.tobytes()

        # Calculate buffer layout
        vertex_offset = 0
        vertex_length = len(vertex_data)
        uv_offset = vertex_length
        uv_length = len(uv_data)
        joint_indices_offset = uv_offset + uv_length
        joint_indices_length = len(joint_indices_data)
        joint_weights_offset = joint_indices_offset + joint_indices_length
        joint_weights_length = len(joint_weights_data)
        index_offset = joint_weights_offset + joint_weights_length
        index_length = len(index_data)
        inverse_bind_matrices_offset = index_offset + index_length
        inverse_bind_matrices_length = len(inverse_bind_matrices_data)

        # Create base binary buffer
        buffer_data = (vertex_data + uv_data + joint_indices_data +
                      joint_weights_data + index_data + inverse_bind_matrices_data)
        total_buffer_length = (vertex_length + uv_length + joint_indices_length +
                              joint_weights_length + index_length + inverse_bind_matrices_length)

        # Add animation data if present
        animation_offsets = []
        if all_animations.get('has_animations', False):
            current_offset = total_buffer_length
            
            for animation in all_animations['animations']:
                # Calculate offsets for this animation
                offsets = {
                    'time_offset': current_offset,
                    'time_length': len(animation['time_data']),
                    'hip_trans_offset': current_offset + len(animation['time_data']),
                    'hip_trans_length': len(animation['hip_translation_data']),
                    'torso_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']),
                    'torso_rot_length': len(animation['torso_rotation_data']),
                    'head_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']) + len(animation['torso_rotation_data']),
                    'head_rot_length': len(animation['head_rotation_data']),
                    'right_leg_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']) + len(animation['torso_rotation_data']) + len(animation['head_rotation_data']),
                    'right_leg_rot_length': len(animation['right_leg_rotation_data']),
                    'left_leg_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']) + len(animation['torso_rotation_data']) + len(animation['head_rotation_data']) + len(animation['right_leg_rotation_data']),
                    'left_leg_rot_length': len(animation['left_leg_rotation_data']),
                    'right_arm_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']) + len(animation['torso_rotation_data']) + len(animation['head_rotation_data']) + len(animation['right_leg_rotation_data']) + len(animation['left_leg_rotation_data']),
                    'right_arm_rot_length': len(animation['right_arm_rotation_data']),
                    'left_arm_rot_offset': current_offset + len(animation['time_data']) + len(animation['hip_translation_data']) + len(animation['torso_rotation_data']) + len(animation['head_rotation_data']) + len(animation['right_leg_rotation_data']) + len(animation['left_leg_rotation_data']) + len(animation['right_arm_rotation_data']),
                    'left_arm_rot_length': len(animation['left_arm_rotation_data'])
                }
                
                animation_offsets.append(offsets)
                
                # Append animation data to buffer
                buffer_data += (animation['time_data'] + animation['hip_translation_data'] +
                               animation['torso_rotation_data'] + animation['head_rotation_data'] + 
                               animation['right_leg_rotation_data'] + animation['left_leg_rotation_data'] + 
                               animation['right_arm_rotation_data'] + animation['left_arm_rotation_data'])

                # Update current offset for next animation
                animation_length = (offsets['time_length'] + offsets['hip_trans_length'] + offsets['torso_rot_length'] + 
                                  offsets['head_rot_length'] + offsets['right_leg_rot_length'] + offsets['left_leg_rot_length'] + 
                                  offsets['right_arm_rot_length'] + offsets['left_arm_rot_length'])
                current_offset += animation_length
                total_buffer_length += animation_length

        buffer_uri = f"data:application/octet-stream;base64,{base64.b64encode(buffer_data).decode()}"

        # Calculate bounding box
        min_pos = vertices.min(axis=0).tolist()
        max_pos = vertices.max(axis=0).tolist()

        # Create joint nodes based on skeleton
        joint_nodes = []
        for i, bone in enumerate(self.skeleton):
            # Calculate relative position to parent
            if bone['parent'] is not None:
                # Find parent bone
                parent_bone = None
                for parent in self.skeleton:
                    if parent['name'] == bone['parent']:
                        parent_bone = parent
                        break

                if parent_bone:
                    # Calculate relative position (child head - parent head)
                    rel_pos = [
                        bone['head'][0] - parent_bone['head'][0],
                        bone['head'][1] - parent_bone['head'][1],
                        bone['head'][2] - parent_bone['head'][2]
                    ]
                    scaled_pos = [pos * self.scale_factor for pos in rel_pos]
                else:
                    # Fallback to absolute if parent not found
                    scaled_pos = [pos * self.scale_factor for pos in bone['head']]
            else:
                # Root bone uses absolute position
                scaled_pos = [pos * self.scale_factor for pos in bone['head']]

            joint_nodes.append({
                "name": bone['name'],
                "translation": scaled_pos
            })

        # Build parent-child relationships
        for i, bone in enumerate(self.skeleton):
            if bone['parent'] is not None:
                parent_idx = self.bone_to_index[bone['parent']]
                # Convert skeleton indices to GLTF node indices (add 1 because root is at index 0)
                child_node_idx = i + 1

                # Add children array to parent if it doesn't exist
                if 'children' not in joint_nodes[parent_idx]:
                    joint_nodes[parent_idx]['children'] = []
                joint_nodes[parent_idx]['children'].append(child_node_idx)

        # Find the root bone (bone with no parent) to connect to the mesh root
        root_bone_idx = None
        for i, bone in enumerate(self.skeleton):
            if bone['parent'] is None:
                root_bone_idx = i + 1  # Add 1 because root mesh node is at index 0
                break

        # Create root node that contains the mesh and connects to skeleton root
        root_node = {
            "name": "Root",
            "mesh": 0,
            "skin": 0,
            "children": [root_bone_idx] if root_bone_idx else []
        }

        # Combine all nodes: [root_node] + joint_nodes
        all_nodes = [root_node] + joint_nodes

        # Create joint list (indices of joint nodes in the nodes array)
        joints = list(range(1, len(joint_nodes) + 1))

        # Create skin
        skin = {
            "inverseBindMatrices": 5,  # Accessor index for inverse bind matrices
            "joints": joints
        }

        # Create base GLTF structure
        gltf = {
            "asset": {
                "version": "2.0",
                "generator": "Minecraft Character Generator"
            },
            "scene": 0,
            "scenes": [{"nodes": [0]}],  # Root node
            "nodes": all_nodes,
            "meshes": [{
                "primitives": [{
                    "attributes": {
                        "POSITION": 0,
                        "TEXCOORD_0": 1,
                        "JOINTS_0": 2,
                        "WEIGHTS_0": 3
                    },
                    "indices": 4,
                    "mode": 4,
                    "material": 0
                }]
            }],
            "skins": [skin],
            "materials": [{
                "name": "MinecraftSkin",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0
                },
                "alphaMode": "BLEND"
            }],
            "textures": [{"source": 0}],
            "images": [{
                "name": "skin",
                "uri": os.path.basename(texture_path)
            }]
        }

        # Add animations if present
        if all_animations.get('has_animations', False):
            gltf_animations = []
            accessor_index = 6  # Start after the main mesh accessors (0-5)
            
            for i, animation in enumerate(all_animations['animations']):
                # Get interpolation type from JSON, default to LINEAR if not specified
                interpolation = animation.get('interpolation', 'LINEAR')
                
                animation_gltf = {
                    "name": animation['display_name'],
                    "samplers": [
                        {"input": accessor_index, "output": accessor_index + 1, "interpolation": interpolation},  # Hip translation
                        {"input": accessor_index, "output": accessor_index + 2, "interpolation": interpolation},  # Torso rotation
                        {"input": accessor_index, "output": accessor_index + 3, "interpolation": interpolation},  # Head rotation
                        {"input": accessor_index, "output": accessor_index + 4, "interpolation": interpolation},  # Right leg rotation
                        {"input": accessor_index, "output": accessor_index + 5, "interpolation": interpolation},  # Left leg rotation
                        {"input": accessor_index, "output": accessor_index + 6, "interpolation": interpolation},  # Right arm rotation
                        {"input": accessor_index, "output": accessor_index + 7, "interpolation": interpolation}   # Left arm rotation
                    ],
                    "channels": [
                        {"sampler": 0, "target": {"node": joints[self.bone_to_index['hip']], "path": "translation"}},
                        {"sampler": 1, "target": {"node": joints[self.bone_to_index['torso']], "path": "rotation"}},
                        {"sampler": 2, "target": {"node": joints[self.bone_to_index['head']], "path": "rotation"}},
                        {"sampler": 3, "target": {"node": joints[self.bone_to_index['right_leg']], "path": "rotation"}},
                        {"sampler": 4, "target": {"node": joints[self.bone_to_index['left_leg']], "path": "rotation"}},
                        {"sampler": 5, "target": {"node": joints[self.bone_to_index['right_arm']], "path": "rotation"}},
                        {"sampler": 6, "target": {"node": joints[self.bone_to_index['left_arm']], "path": "rotation"}}
                    ]
                }
                gltf_animations.append(animation_gltf)
                
                # Move to the next set of accessors (8 accessors per animation: 1 time + 7 outputs)
                accessor_index += 8
            
            gltf["animations"] = gltf_animations

        # Add accessors for mesh data
        gltf["accessors"] = [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(vertices),
                "type": "VEC3",
                "max": max_pos,
                "min": min_pos
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": len(uvs),
                "type": "VEC2"
            },
            {
                "bufferView": 2,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(joint_indices),
                "type": "VEC4"
            },
            {
                "bufferView": 3,
                "componentType": 5126,  # FLOAT
                "count": len(joint_weights),
                "type": "VEC4"
            },
            {
                "bufferView": 4,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(indices),
                "type": "SCALAR"
            },
            {
                "bufferView": 5,
                "componentType": 5126,  # FLOAT
                "count": len(self.skeleton),
                "type": "MAT4"
            }
        ]

        # Add animation accessors if present
        if all_animations.get('has_animations', False):
            buffer_view_index = 6  # Start after the main mesh buffer views (0-5)
            
            for animation in all_animations['animations']:
                gltf["accessors"].extend([
                    {
                        "bufferView": buffer_view_index,
                        "componentType": 5126,  # FLOAT
                        "count": animation['keyframe_count'],  # Time data always uses original keyframe count
                        "type": "SCALAR",
                        "max": [animation['duration']],
                        "min": [0.0]
                    },
                    {
                        "bufferView": buffer_view_index + 1,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],  # Use accessor count for animation data
                        "type": "VEC3"
                    },
                    {
                        "bufferView": buffer_view_index + 2,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    },
                    {
                        "bufferView": buffer_view_index + 3,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    },
                    {
                        "bufferView": buffer_view_index + 4,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    },
                    {
                        "bufferView": buffer_view_index + 5,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    },
                    {
                        "bufferView": buffer_view_index + 6,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    },
                    {
                        "bufferView": buffer_view_index + 7,
                        "componentType": 5126,  # FLOAT
                        "count": animation['accessor_keyframe_count'],
                        "type": "VEC4"
                    }
                ])
                
                # Move to next set of buffer views (8 buffer views per animation)
                buffer_view_index += 8

        # Add buffer views for mesh data
        gltf["bufferViews"] = [
            {
                "buffer": 0,
                "byteOffset": vertex_offset,
                "byteLength": vertex_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": uv_offset,
                "byteLength": uv_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": joint_indices_offset,
                "byteLength": joint_indices_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": joint_weights_offset,
                "byteLength": joint_weights_length,
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": index_offset,
                "byteLength": index_length,
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": inverse_bind_matrices_offset,
                "byteLength": inverse_bind_matrices_length
            }
        ]

        # Add animation buffer views if present
        if all_animations.get('has_animations', False):
            for offsets in animation_offsets:
                gltf["bufferViews"].extend([
                    {
                        "buffer": 0,
                        "byteOffset": offsets['time_offset'],
                        "byteLength": offsets['time_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['hip_trans_offset'],
                        "byteLength": offsets['hip_trans_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['torso_rot_offset'],
                        "byteLength": offsets['torso_rot_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['head_rot_offset'],
                        "byteLength": offsets['head_rot_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['right_leg_rot_offset'],
                        "byteLength": offsets['right_leg_rot_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['left_leg_rot_offset'],
                        "byteLength": offsets['left_leg_rot_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['right_arm_rot_offset'],
                        "byteLength": offsets['right_arm_rot_length']
                    },
                    {
                        "buffer": 0,
                        "byteOffset": offsets['left_arm_rot_offset'],
                        "byteLength": offsets['left_arm_rot_length']
                    }
                ])

        # Add buffer
        gltf["buffers"] = [{
            "byteLength": total_buffer_length,
            "uri": buffer_uri
        }]

        # Write GLTF file
        with open(output_path, 'w') as f:
            json.dump(gltf, f, indent=2)

        print(f"Generated GLTF file: {output_path}")
        print(f"  Vertices: {len(vertices)}")
        print(f"  Triangles: {len(indices) // 3}")
        print(f"  Joints: {len(self.skeleton)}")
        print(f"  Scale factor: {self.scale_factor}")
        print(f"  Body parts have 0.01 unit gaps to prevent face colliding")
        print(f"  Overlay meshes scaled 1.1x from center to avoid face clashing")
        print(f"  Texel-to-mesh mapping ensures square aspect ratios")
        print(f"  Skeletal animation with proper skinning enabled")
        if all_animations.get('has_animations', False):
            print(f"  Animations: {len(all_animations['animations'])} defined")
            for animation in all_animations['animations']:
                print(f"    - {animation['display_name']} ({animation['duration']}s cycle, {animation['keyframe_count']} keyframes)")
        else:
            print(f"  No animations defined in specification")

def main():
    if len(sys.argv) < 3:
        print("Usage: python minecraft_character_generator.py <texture_path> <output_gltf> [spec_file] [scale_factor]")
        print("  spec_file: minecraft_character.json (classic/Steve) or minecraft_character_slim.json (slim/Alex)")
        print("  scale_factor: Scale factor for the entire model (default: 1.0)")
        print("Example: python minecraft_character_generator.py ../data/textures/pink-smoothie.png minecraft_character_pink_smoothie.gltf")
        print("Example: python minecraft_character_generator.py ../data/textures/pink-smoothie.png minecraft_character_slim_pink_smoothie.gltf minecraft_character_slim.json")
        print("Example: python minecraft_character_generator.py ../data/textures/pink-smoothie.png minecraft_character_scaled.gltf minecraft_character.json 0.5")
        sys.exit(1)

    texture_path = sys.argv[1]
    output_path = sys.argv[2]
    spec_file = sys.argv[3] if len(sys.argv) > 3 else 'minecraft_character.json'

    # If scale factor not provided, use the constructor's default
    if len(sys.argv) > 4:
        scale_factor = float(sys.argv[4])
        generator = MinecraftCharacterGenerator(spec_file, scale_factor)
    else:
        generator = MinecraftCharacterGenerator(spec_file)
        scale_factor = generator.scale_factor
    print(f"Using character specification: {spec_file}")
    print(f"Scale factor: {scale_factor}")

    # Generate mesh
    mesh = generator.generate_mesh(texture_path)

    # Export to GLTF
    generator.mesh_to_gltf(mesh, output_path, texture_path)

if __name__ == '__main__':
    main()