#!/usr/bin/env python3
"""
Script to generate BlockType.h from blocks.yaml
This ensures the enum is always in sync with the YAML definition
"""

import yaml
import os
import sys
from pathlib import Path

def generate_block_types_header(yaml_file_path, output_header_path):
    """Generate BlockType.h from blocks.yaml"""
    
    # Read the YAML file
    try:
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading YAML file {yaml_file_path}: {e}")
        return False
    
    if 'blocks' not in data:
        print("Error: 'blocks' key not found in YAML file")
        return False
    
    blocks = data['blocks']
    
    # Sort blocks by ID to ensure consistent enum ordering
    blocks.sort(key=lambda x: x['id'])
    
    # Find the Test1 block ID to determine instanced vs uninstanced boundary
    test1_id = None
    for block in blocks:
        if block.get('type') == 'BlockTypeTest1':
            test1_id = block['id']
            break
    
    if test1_id is None:
        print("Error: BlockTypeTest1 not found in blocks.yaml")
        return False
    
    # Generate the header content
    header_content = f"""#pragma once
// THIS FILE IS AUTO-GENERATED FROM data/assets/blocks.yaml
// DO NOT EDIT MANUALLY - YOUR CHANGES WILL BE OVERWRITTEN
// To modify block types, edit data/assets/blocks.yaml and rebuild

enum BlockType
{{
"""
    
    # Add enum values
    for block in blocks:
        block_type = block.get('type', '')
        comment = f"    {block_type},  // ID: {block['id']} - {block.get('name', 'Unknown')}"
        header_content += comment + "\n"
    
    # Add BlockTypeNum
    max_id = max(block['id'] for block in blocks)
    header_content += f"\n    BlockTypeNum = {max_id + 1},\n"
    header_content += "};\n"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)
    
    # Write the header file
    try:
        with open(output_header_path, 'w') as file:
            file.write(header_content)
        print(f"Successfully generated {output_header_path}")
        return True
    except Exception as e:
        print(f"Error writing header file {output_header_path}: {e}")
        return False

def main():
    # Default paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    yaml_file = project_root / "data" / "assets" / "blocks.yaml"
    output_header = project_root / "generated" / "voxelengine" / "BlockType.h"
    
    # Allow command line override
    if len(sys.argv) >= 2:
        yaml_file = Path(sys.argv[1])
    if len(sys.argv) >= 3:
        output_header = Path(sys.argv[2])
    
    if not yaml_file.exists():
        print(f"Error: YAML file not found: {yaml_file}")
        sys.exit(1)
    
    success = generate_block_types_header(yaml_file, output_header)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()