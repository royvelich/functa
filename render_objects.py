# standard library
from typing import Tuple, List, Protocol, Optional
import argparse
from pathlib import Path
import shutil
import os
import subprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str)
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--categories', type=str, nargs='+')
    args = parser.parse_args()

    for category in args.categories:
        output_folder_path = Path(args.output_path)
        input_folder_path = Path(args.input_path)
        input_category_folder_path = input_folder_path / category
        output_category_folder_path = output_folder_path / category
        output_category_folder_path.mkdir(parents=True, exist_ok=True)
        files = [item for item in input_category_folder_path.iterdir() if item.is_file()]
        for file in files:
            command = [
                'blender', '-b', '-P', './external/objaverse-rendering/scripts/blender_script.py', '--',
                '--object_path', str(file),
                '--output_dir', str(output_category_folder_path),
                '--engine', 'CYCLES',
                '--num_images', '12',
                '--camera_dist', '2']

            result = subprocess.run(command, text=True, capture_output=True)
            pass
