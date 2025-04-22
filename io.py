import csv
import fnmatch
import gzip
import inspect
import json
import lzma
import multiprocessing
import os
import pickle
import shutil
from typing import Dict, List, Union


def create_dir(dir, print_info=False, print_func=print, suppress_errors=False) -> bool:
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)
            if print_info:
                print_func(f"Created dir: {dir}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print_func(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def delete_file_or_dir(path, print_info=False, print_func=print, suppress_errors=False) -> bool:
    try:
        if os.path.exists(path):
            is_file = os.path.isfile(path)
            if is_file:
                os.remove(path)
            else:
                shutil.rmtree(path)
            if print_info:
                print_func(f"Deleted {'file' if is_file else 'dir'}: {path}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print_func(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def copy_file_or_dir(source_path, target_dir, print_info=False, print_func=print, suppress_errors=False) -> bool:
    try:
        if os.path.exists(source_path):
            create_dir(target_dir, suppress_errors=suppress_errors)
            is_file = os.path.isfile(source_path)
            if is_file:
                shutil.copy2(source_path, target_dir)
            else:
                shutil.copytree(source_path, os.path.join(target_dir, os.path.basename(source_path)), dirs_exist_ok=True)
            if print_info:
                print_func(f"Copied {'file' if is_file else 'dir'}: {source_path} -> {os.path.join(target_dir, os.path.basename(source_path))}")
            return True
        else:
            return False
    except Exception as e:
        if suppress_errors:
            print_func(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def move_file_or_dir(source_path, target_dir, overwrite_existed=False, print_info=False, print_func=print, suppress_errors=False) -> bool:
    try:
        if os.path.exists(source_path):
            source_is_file = os.path.isfile(source_path)

            create_dir(target_dir, suppress_errors=suppress_errors)
            target_path = os.path.join(target_dir, os.path.basename(source_path))

            if os.path.exists(target_path):
                target_is_file = os.path.isfile(target_path)
                if not overwrite_existed:
                    if print_info:
                        print_func(f"Target path {target_path} already exists and `overwrite_existed=False`. Skipping move.")
                    return False

                elif source_is_file ^ target_is_file:  # source and target are different types
                    if print_info:
                        print_func(f"Source and target are different types. `source_is_file={source_is_file}` but `target_is_file={target_is_file}`. Skipping move.")
                    return False

                elif target_is_file:  # both source and target are files, delete then remove
                    delete_file_or_dir(target_path, print_info=print_info, suppress_errors=suppress_errors)
                    shutil.move(source_path, target_dir)
                    if print_info:
                        print_func(f"Overwritten file: {source_path} -> {os.path.join(target_dir, os.path.basename(source_path))}")
                    return True

                else:  # both source and target are directories, merge the source and target
                    all_results = []
                    for subpath in os.listdir(source_path):
                        source_subpath = os.path.join(source_path, subpath)
                        result = move_file_or_dir(source_subpath, target_path, overwrite_existed=overwrite_existed, print_info=print_info, suppress_errors=suppress_errors)
                        all_results.append(result)
                    success = all(all_results)
                    if success:
                        delete_file_or_dir(source_path, print_info=print_info, suppress_errors=suppress_errors)
                    return all(all_results)

            else:
                shutil.move(source_path, target_dir)
                if print_info:
                    print_func(f"Moved {'file' if source_is_file else 'dir'}: {source_path} -> {os.path.join(target_dir, os.path.basename(source_path))}")
                return True

        else:
            return False

    except Exception as e:
        if suppress_errors:
            print_func(f"Exception within `{inspect.currentframe().f_code.co_name}`: {e}")
            return False
        else:
            raise e


def find_files(dir, name_pattern):
    """
    Search for files matching a specified pattern in a given directory and its subdirectories.

    Args:
    - dir: String of root directory path to search.
    - name_pattern: String of pattern to match filename against (e.g. '*.txt' to match all txt files).

    Returns:
    - A list of full paths to the found files.
    """
    matches = []
    for root, dirs, files in os.walk(dir):
        for filename in fnmatch.filter(files, name_pattern):
            matches.append(os.path.join(root, filename))
    return matches


def save_compressed_file_7z(data, file_path):
    create_dir(os.path.dirname(file_path), suppress_errors=True)
    with lzma.open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_compressed_file_7z(file_path):
    with lzma.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def save_compressed_file_gz(data, file_path, compresslevel: int = 9):
    create_dir(os.path.dirname(file_path), suppress_errors=True)
    with gzip.open(file_path, "wb", compresslevel=compresslevel) as file:
        pickle.dump(data, file)


def load_compressed_file_gz(file_path):
    with gzip.open(file_path, "rb") as file:
        data = pickle.load(file)
    return data


def read_csv(file_path, has_header=True) -> Union[List[List], List[Dict]]:
    """
    Read a CSV file and return its content.

    Args:
    - file_path (str): Path to the CSV file.
    - has_header (bool): Whether the CSV file has a header. Default is True.

    Returns:
    - list of list or dict: Content of the CSV file.
      If has_header is True, return a list of dictionaries;
      if has_header is False, return a list of lists.
    """
    data = []
    with open(file_path, newline='', encoding='utf-8') as f:
        if has_header:
            csvreader = csv.DictReader(f)
            for row in csvreader:
                data.append(dict(row))
        else:
            csvreader = csv.reader(f)
            for row in csvreader:
                data.append(row)
    return data


def load_json(file_path):
    with open(file_path, "r", encoding="utf8") as f:
        data = json.load(f)
    return data


def save_json(data, file_path, indent=4, **kwargs):
    create_dir(os.path.dirname(file_path), suppress_errors=True)
    with open(file_path, "w", encoding="utf8") as f:
        f.write(f"{json.dumps(data, ensure_ascii=False, indent=indent, **kwargs)}\n")


def load_jsonl(file_path) -> List:
    data = []
    with open(file_path, "r", encoding="utf8") as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding line: {line}")
                continue
    return data


def save_jsonl(data, file_path, **kwargs):
    create_dir(os.path.dirname(file_path), suppress_errors=True)
    with open(file_path, "w", encoding="utf8") as f:
        for ins in data:
            f.write(f"{json.dumps(ins, ensure_ascii=False, **kwargs)}\n")


def _compress_image(image_path, print_info=False, print_func=print):
    import cv2
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    cv2.imwrite(image_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if print_info:
        print_func(f'Compression done for "{image_path}".')


def compress_png_image(image_path, non_block=True, print_info=False, print_func=print):
    if non_block:  # Start a new process to run the compression
        process = multiprocessing.Process(target=_compress_image, args=(image_path, print_info, print_func))
        process.start()
    else:
        _compress_image(image_path, print_info, print_func)
