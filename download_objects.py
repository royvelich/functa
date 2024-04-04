# standard library
from typing import Tuple, List, Protocol, Optional
import argparse
from pathlib import Path
import shutil
import os

# objaverse
import objaverse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', type=str)
    parser.add_argument('--categories', type=str, nargs='+')
    parser.add_argument('--download-processes', type=int)
    args = parser.parse_args()

    # reference - https://colab.research.google.com/drive/1ZLA4QufsiI_RuNlamKqV7D7mn40FbWoY#scrollTo=SlIEPsu6H4iW
    lvis_annotations = objaverse.load_lvis_annotations()

    for category in args.categories:
        objaverse_objects = objaverse.load_objects(uids=lvis_annotations[category], download_processes=args.download_processes)
        for objaverse_object in objaverse_objects.values():
            downloaded_file_path = Path(objaverse_object)
            output_folder_path = Path(args.output_path)
            output_file_path = output_folder_path / Path(category) / downloaded_file_path.name
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src=str(downloaded_file_path), dst=str(output_file_path))