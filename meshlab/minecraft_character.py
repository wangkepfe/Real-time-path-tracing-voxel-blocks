import os
import json
import struct
import math
import numpy as np
import traceback

def create_gltf_model():
    """Create a GLTF model of a Minecraft character"""
    print("Creating GLTF model...")

    # Define the basic box geometry for a Minecraft character
    # Each body part is a box with specific dimensions

    # Minecraft character proportions (in pixels, scaled to meters)
    scale = 0.01  # 1 pixel = 1 cm

    # Define body parts with their dimensions and positions
    body_parts = {
        "head": {
            "size": [8, 8, 8],
            "position": [0, 22, 0],
            "uv_offset": [0, 0]
        },
        "body": {
            "size": [8, 12, 4],
            "position": [0, 10, 0],
            "uv_offset": [16, 16]
        },
        "left_arm": {
            "size": [4, 12, 4],
            "position": [-6, 10, 0],
            "uv_offset": [32, 48]
        },
        "right_arm": {
            "size": [4, 12, 4],
            "position": [6, 10, 0],
            "uv_offset": [40, 16]
        },
        "left_leg": {
            "size": [4, 12, 4],
            "position": [-2, -2, 0],
            "uv_offset": [16, 48]
        },
        "right_leg": {
            "size": [4, 12, 4],
            "position": [2, -2, 0],
            "uv_offset": [0, 16]
        }
    }

    # Create vertices, indices, and texture coordinates for all body parts
    all_vertices = []
    all_indices = []
    all_texcoords = []
    vertex_offset = 0

    for part_name, part_data in body_parts.items():
        size = part_data["size"]
        pos = part_data["position"]
        uv_offset = part_data["uv_offset"]

        # Create a box with the specified dimensions
        w, h, d = size[0] * scale, size[1] * scale, size[2] * scale
        x, y, z = pos[0] * scale, pos[1] * scale, pos[2] * scale

        # Define the 8 vertices of the box
        box_vertices = [
            [x - w/2, y - h/2, z - d/2],  # 0: left, bottom, back
            [x + w/2, y - h/2, z - d/2],  # 1: right, bottom, back
            [x + w/2, y + h/2, z - d/2],  # 2: right, top, back
            [x - w/2, y + h/2, z - d/2],  # 3: left, top, back
            [x - w/2, y - h/2, z + d/2],  # 4: left, bottom, front
            [x + w/2, y - h/2, z + d/2],  # 5: right, bottom, front
            [x + w/2, y + h/2, z + d/2],  # 6: right, top, front
            [x - w/2, y + h/2, z + d/2],  # 7: left, top, front
        ]

        # Define the 12 triangles (2 per face, 6 faces)
        # Using counter-clockwise winding for outward-facing normals
        box_indices = [
            # Front face (z + d/2) - vertices 4, 5, 6, 7
            [4, 5, 6], [6, 7, 4],
            # Back face (z - d/2) - vertices 0, 1, 2, 3
            [0, 3, 2], [2, 1, 0],
            # Left face (x - w/2) - vertices 0, 3, 4, 7
            [0, 4, 7], [7, 3, 0],
            # Right face (x + w/2) - vertices 1, 2, 5, 6
            [1, 2, 6], [6, 5, 1],
            # Top face (y + h/2) - vertices 2, 3, 6, 7
            [2, 3, 7], [7, 6, 2],
            # Bottom face (y - h/2) - vertices 0, 1, 4, 5
            [0, 1, 5], [5, 4, 0]
        ]

        # Simple texture coordinates for each vertex
        # This is a simplified version - in reality you'd map to the Minecraft skin texture
        box_texcoords = [
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # Back face
            [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]   # Front face
        ]

        # Add vertices to the global list
        all_vertices.extend(box_vertices)
        all_texcoords.extend(box_texcoords)

        # Add indices to the global list (adjusting for vertex offset)
        for triangle in box_indices:
            all_indices.extend([vertex_offset + i for i in triangle])

        vertex_offset += len(box_vertices)

    # Create the GLTF structure
    gltf = {
        "asset": {
            "version": "2.0",
            "generator": "Minecraft Character Generator"
        },
        "scene": 0,
        "scenes": [
            {
                "nodes": [0]
            }
        ],
        "nodes": [
            {
                "mesh": 0,
                "name": "MinecraftCharacter"
            }
        ],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "TEXCOORD_0": 1
                        },
                        "indices": 2,
                        "mode": 4  # TRIANGLES
                    }
                ]
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(all_vertices),
                "type": "VEC3",
                "max": [max(v[0] for v in all_vertices), max(v[1] for v in all_vertices), max(v[2] for v in all_vertices)],
                "min": [min(v[0] for v in all_vertices), min(v[1] for v in all_vertices), min(v[2] for v in all_vertices)]
            },
            {
                "bufferView": 1,
                "componentType": 5126,  # FLOAT
                "count": len(all_texcoords),
                "type": "VEC2"
            },
            {
                "bufferView": 2,
                "componentType": 5123,  # UNSIGNED_SHORT
                "count": len(all_indices),
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(all_vertices) * 3 * 4,  # 3 floats per vertex, 4 bytes per float
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(all_vertices) * 3 * 4,
                "byteLength": len(all_texcoords) * 2 * 4,  # 2 floats per texcoord, 4 bytes per float
                "target": 34962  # ARRAY_BUFFER
            },
            {
                "buffer": 0,
                "byteOffset": len(all_vertices) * 3 * 4 + len(all_texcoords) * 2 * 4,
                "byteLength": len(all_indices) * 2,  # 2 bytes per short
                "target": 34963  # ELEMENT_ARRAY_BUFFER
            }
        ],
        "buffers": [
            {
                "byteLength": len(all_vertices) * 3 * 4 + len(all_texcoords) * 2 * 4 + len(all_indices) * 2,
                "uri": "minecraft_char.bin"
            }
        ]
    }

    return gltf, all_vertices, all_texcoords, all_indices

def create_binary_data(vertices, texcoords, indices):
    """Create binary data for GLTF"""
    binary_data = bytearray()

    # Add vertex data
    for vertex in vertices:
        for component in vertex:
            binary_data.extend(struct.pack('<f', component))

    # Add texture coordinate data
    for texcoord in texcoords:
        for component in texcoord:
            binary_data.extend(struct.pack('<f', component))

    # Add index data
    for index in indices:
        binary_data.extend(struct.pack('<H', index))

    return binary_data

def main():
    try:
        print("Starting Minecraft character GLTF generation...")

        # Create the GLTF model
        gltf, vertices, texcoords, indices = create_gltf_model()

        # Create binary data
        binary_data = create_binary_data(vertices, texcoords, indices)

        # Write GLTF file
        with open('../data/models/minecraft_char.gltf', 'w') as f:
            json.dump(gltf, f, indent=2)

        # Write binary data
        with open('../data/models/minecraft_char.bin', 'wb') as f:
            f.write(binary_data)

        print("Successfully created minecraft_char.gltf and minecraft_char.bin")
        print(f"Generated {len(vertices)} vertices and {len(indices)} indices")

    except Exception as e:
        print(f"Error during GLTF generation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()