"""Multi-control dataset utilities. Supports variable controls per image."""

from pathlib import Path
from collections import defaultdict
import torch
import numpy as np


def detect_control_paths(directory_config):
    """Detect all control folders from base control_path, handling any naming convention."""
    control_paths = []
    if 'control_path' not in directory_config:
        return control_paths

    base_control = Path(directory_config['control_path'])
    if not base_control.exists() or not base_control.is_dir():
        return control_paths

    control_paths.append(base_control)
    parent_dir = base_control.parent

    # Extract the base name and trailing number from the control folder name
    base_name = base_control.name
    # Find where the trailing digits start
    trailing_digit_idx = len(base_name)
    while trailing_digit_idx > 0 and base_name[trailing_digit_idx - 1].isdigit():
        trailing_digit_idx -= 1

    prefix = base_name[:trailing_digit_idx]
    num_str = base_name[trailing_digit_idx:]

    # Determine the starting number for the next control folder
    start_num = int(num_str) + 1 if num_str else 2

    # Look for sequential control folders (up to 4 more)
    for i in range(start_num, start_num + 4):
        additional_control = parent_dir / f'{prefix}{i}'
        if additional_control.exists() and additional_control.is_dir():
            control_paths.append(additional_control)

    return control_paths


def build_control_file_stems_list(control_paths):
    """Build file stem mappings for each control folder."""
    control_file_stems_list = []
    for control_path in control_paths:
        stems = {path.stem: path for path in control_path.glob('*') if path.is_file()}
        control_file_stems_list.append(stems)
    return control_file_stems_list


def process_control_files_for_image(image_file, control_file_stems_list, control_files_lists):
    """Add control file paths for image, storing None for missing controls."""
    if len(control_file_stems_list) == 0:
        return

    for i, control_stems in enumerate(control_file_stems_list):
        if image_file.stem in control_stems:
            control_files_lists[i].append(str(control_stems[image_file.stem]))
        else:
            # Control file is optional - store None if missing
            control_files_lists[i].append(None)


def count_controls_in_example(example, max_controls=5):
    """Return (num_controls, has_any_controls) for metadata example."""
    num_controls = 0
    for i in range(1, max_controls + 1):
        key = 'control_file' if i == 1 else f'control_file_{i}'
        if key in example:
            num_controls += 1
        else:
            break
    return num_controls, num_controls > 0


def create_empty_return_dict(control_paths):
    """Create empty return dict with keys for all control types."""
    empty_return = {
        'image_spec': [],
        'mask_file': [],
        'caption': [],
        'ar_bucket': [],
        'size_bucket': [],
        'is_video': []
    }
    for i in range(len(control_paths)):
        control_num = i + 1
        key = 'control_file' if control_num == 1 else f'control_file_{control_num}'
        empty_return[key] = []
    return empty_return


def add_control_files_to_return(ret, example, control_paths):
    """Add control file entries from example to return dict."""
    for i in range(len(control_paths)):
        control_num = i + 1
        key = 'control_file' if control_num == 1 else f'control_file_{control_num}'
        ret[key] = [example[key][0]]


def stack_control_tensors_with_padding(control_tensors_lists, caching_batch_size, batch_start_idx):
    """Stack control tensors, padding with zeros for missing controls."""
    c_tensors = []
    for ctl_list in control_tensors_lists:
        batch_slice = ctl_list[batch_start_idx:batch_start_idx + caching_batch_size]

        # Check if all items in this batch are None
        if all(t is None for t in batch_slice):
            c_tensors.append(None)
        else:
            # Stack available tensors, replacing None with zero-tensors
            tensors_to_stack = []
            for t in batch_slice:
                if t is not None:
                    tensors_to_stack.append(t[0])
                else:
                    tensors_to_stack.append(None)

            # Find first valid tensor as reference for shape
            valid_tensors = [t for t in tensors_to_stack if t is not None]
            if valid_tensors:
                # Pad the batch with zeros where controls are missing
                final_tensors = []
                for t in tensors_to_stack:
                    if t is not None:
                        final_tensors.append(t)
                    else:
                        final_tensors.append(torch.zeros_like(valid_tensors[0]))
                c_tensor = torch.stack(final_tensors)
                c_tensors.append(c_tensor)
            else:
                c_tensors.append(None)

    return c_tensors


def process_control_file_for_latents(control_file, preprocess_media_file_fn, size_bucket):
    """Process control file, returning None if control is missing for this image."""
    if control_file is None:
        return None

    control_items = preprocess_media_file_fn((None, control_file), None, size_bucket)
    assert len(control_items) == 1, f"Expected 1 item from preprocess, got {len(control_items)}"
    return control_items[0]


def filter_control_tensors(control_tensors):
    """Filter out None control tensors."""
    return [t for t in control_tensors if t is not None]


def get_control_result_keys(control_tensor_idx, num_active_controls):
    """Get result key for control latents (e.g., 'control_latents' or 'control_latents_2')."""
    if control_tensor_idx == 0:
        return 'control_latents'
    else:
        return f'control_latents_{control_tensor_idx + 1}'
