import json
from pathlib import Path
import base64
from typing import List

import cv2
import numpy as np

from cv2.typing import MatLike

def content_from_llm_block(resp: str):
    if resp.endswith('```'):
        if resp.startswith('```markdown'):
            resp = resp[8:-3].strip()
        elif resp.startswith('```'):
            resp = resp[3:-3].strip()
        else:
            raise ValueError(f"Invalid response format: {resp}")
    
    return resp

def cv2_to_base64(
        cv_img: np.ndarray,
        img_format: str='jpg',
    ) -> str:
    
    if img_format == 'jpg':
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    elif img_format == 'png':
        encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
    else:
        raise ValueError(f"Unsupported image format: {img_format}")
    
    is_success, buffer = cv2.imencode(f'.{img_format}', cv_img, encode_param)

    if not is_success:
        raise ValueError("Failed to encode image")
    
    b64_img = base64.b64encode(buffer.tobytes()).decode('utf-8')

    return b64_img

def split_pic(
        img_path: Path, 
        min_gap_width: int=8,
        min_slice_height: int=32,
        write_file: bool=False
    ) -> List[MatLike]:
    MAX_VERTICAL_PROJ_VAL = 8

    assert img_path.is_file(), f"File {img_path.absolute()} not found!"
    cv_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

    __, img_thresh = cv2.threshold(cv_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_vertical_proj = np.sum(img_thresh, axis=1)

    split_indices = []
    start_index: int|None = None

    for i, v_proj in enumerate(img_vertical_proj):
        if v_proj < MAX_VERTICAL_PROJ_VAL:
            if not start_index:
                start_index = i
        else:
            if start_index and (i-start_index)>min_gap_width:
                split_indices.append((start_index+i)//2)
            start_index = None

    merge_start_index = 0
    filtered_split_indices: List[int] = []
    for i, idx in enumerate(split_indices + [cv_img.shape[0]]):
        if (idx - merge_start_index) > min_slice_height:
            filtered_split_indices.append(idx)
            merge_start_index = idx
        else:
            continue

    # Log the slice height std
    # print(np.diff(split_indices).std())

    img_segments: List[MatLike] = []
    prev_id = 0

    if write_file:
        Path('./slices').mkdir(exist_ok=True)

    for i, idx in enumerate(filtered_split_indices):
        segment = cv_img[prev_id:idx, :]
        img_segments.append(segment)

        prev_id = idx
        if write_file:
            file_path = Path('./slices') / f'slice-{i}.jpg'
            cv2.imwrite(str(file_path), segment)

    return img_segments


# split_pic(Path('./data/test.jpg'), min_slice_height=1024, write_file=True)