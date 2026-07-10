"""
Per-structure quality-control metrics for segmentation masks.

This module computes descriptive metrics from already-generated masks (and, optionally,
the CT they were derived from) that can be used to spot likely false negatives or
inconsistent segmentations without ground truth: size, shape/compactness, connected
components, boundary contact and intensity distribution. It does not decide what counts
as "wrong" - that requires comparing these numbers against an empirical reference built up
across many cases, which is a separate later step.

Each metric group is its own small function operating on plain numpy arrays, so they can be
reused independently (e.g. from an in-run hook) without going through the file-loading
orchestrator `calculate_mask_metrics`.
"""
from pathlib import Path
from typing import Union

import numpy as np
import nibabel as nib
from scipy import ndimage

from totalsegmentator.statistics import touches_border


def volume_metrics(mask: np.ndarray, spacing: tuple) -> dict:
    """
    num_voxels, volume_mm3 and is_empty for a binary mask.

    spacing: (x, y, z) voxel spacing in mm, as returned by nib header.get_zooms().

    volume_mm3 is in mm3, matching the units of the "volume" field statistics.py writes to
    statistics.json (see get_basic_statistics), so the two can be compared/cross-checked
    directly without a unit conversion.

    Returns: {"num_voxels": int, "volume_mm3": float, "is_empty": int}
    """
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    num_voxels = int(mask.sum())
    return {
        "num_voxels": num_voxels,
        "volume_mm3": round(float(num_voxels * vox_vol), 2),
        "is_empty": int(num_voxels == 0),
    }


def shape_metrics(mask: np.ndarray, spacing: tuple) -> dict:
    """
    Relative centroid/bbox position and mask-to-bbox compactness for a binary mask.

    Centroid and bbox extents are expressed as fractions of the image size along each axis,
    so they are comparable across images of different dimensions. bbox extents use an
    inclusive-max convention (max_idx - min_idx + 1).

    spacing: (x, y, z) voxel spacing in mm.

    bbox_volume_mm3 is in mm3, matching statistics.json's "volume" units (see volume_metrics).

    Returns: {"centroid_x_rel", "centroid_y_rel", "centroid_z_rel", "bbox_x_rel",
        "bbox_y_rel", "bbox_z_rel", "bbox_volume_mm3", "mask_to_bbox_ratio"}
    """
    shape = mask.shape
    idx = np.argwhere(mask)
    if idx.shape[0] == 0:
        return {
            "centroid_x_rel": None, "centroid_y_rel": None, "centroid_z_rel": None,
            "bbox_x_rel": 0.0, "bbox_y_rel": 0.0, "bbox_z_rel": 0.0,
            "bbox_volume_mm3": 0.0, "mask_to_bbox_ratio": 0.0,
        }

    centroid = idx.mean(axis=0)
    mins = idx.min(axis=0)
    maxs = idx.max(axis=0)
    bbox_extent_vox = maxs - mins + 1
    bbox_extent_rel = bbox_extent_vox / np.array(shape)

    vox_vol = spacing[0] * spacing[1] * spacing[2]
    bbox_voxel_count = int(np.prod(bbox_extent_vox))
    num_voxels = int(idx.shape[0])

    return {
        "centroid_x_rel": round(float(centroid[0] / shape[0]), 4),
        "centroid_y_rel": round(float(centroid[1] / shape[1]), 4),
        "centroid_z_rel": round(float(centroid[2] / shape[2]), 4),
        "bbox_x_rel": round(float(bbox_extent_rel[0]), 4),
        "bbox_y_rel": round(float(bbox_extent_rel[1]), 4),
        "bbox_z_rel": round(float(bbox_extent_rel[2]), 4),
        "bbox_volume_mm3": round(float(bbox_voxel_count * vox_vol), 2),
        "mask_to_bbox_ratio": round(float(num_voxels / bbox_voxel_count), 4),
    }


def component_metrics(mask: np.ndarray) -> dict:
    """
    Connected-component structure of a binary mask, via scipy.ndimage.label.

    Returns: {"num_components": int, "largest_component_fraction": float}
    """
    num_voxels = int(mask.sum())
    if num_voxels == 0:
        return {"num_components": 0, "largest_component_fraction": 0.0}

    labeled, num_components = ndimage.label(mask)
    counts = np.bincount(labeled.flatten())[1:]  # exclude background (label 0)
    largest_component_fraction = float(counts.max() / num_voxels)

    return {
        "num_components": int(num_components),
        "largest_component_fraction": round(largest_component_fraction, 4),
    }


def boundary_metrics(mask: np.ndarray) -> dict:
    """
    Whether/how much a binary mask touches the edge of the image volume.

    touches_boundary reuses statistics.touches_border, which only inspects the outer 3
    voxels of each face. boundary_fraction is a finer-grained companion metric: the exact
    fraction of the mask's voxels that sit on any of the 6 outermost faces (single-voxel
    layer), which touches_boundary alone cannot distinguish (e.g. one stray voxel vs. an
    entire face).

    Returns: {"touches_boundary": int, "boundary_fraction": float}
    """
    num_voxels = int(mask.sum())
    if num_voxels == 0:
        return {"touches_boundary": 0, "boundary_fraction": 0.0}

    touches = int(touches_border(mask))

    boundary = np.zeros_like(mask, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True
    boundary_voxels = int(np.count_nonzero(mask & boundary))

    return {
        "touches_boundary": touches,
        "boundary_fraction": round(float(boundary_voxels / num_voxels), 4),
    }


def intensity_metrics(mask: np.ndarray, ct: np.ndarray) -> dict:
    """
    HU intensity distribution of a CT image restricted to a binary mask.

    Returns: {"mean_HU", "median_HU", "std_HU", "p05_HU", "p95_HU"}
    """
    if mask.sum() == 0:
        return {"mean_HU": None, "median_HU": None, "std_HU": None, "p05_HU": None, "p95_HU": None}

    vals = ct[mask]
    return {
        "mean_HU": round(float(np.mean(vals)), 2),
        "median_HU": round(float(np.median(vals)), 2),
        "std_HU": round(float(np.std(vals)), 2),
        "p05_HU": round(float(np.percentile(vals, 5)), 2),
        "p95_HU": round(float(np.percentile(vals, 95)), 2),
    }


_EMPTY_INTENSITY_METRICS = {"mean_HU": None, "median_HU": None, "std_HU": None, "p05_HU": None, "p95_HU": None}


def calculate_mask_metrics(mask_path: Union[str, Path], ct_path: Union[str, Path, None] = None,
                           task: str = "total", class_map: dict = None) -> dict:
    """
    Compute the full set of QC metrics (volume, shape, connected components, boundary
    contact and, if a CT is given, HU intensity) for every structure in mask_path.

    mask_path: either
        - a directory of per-structure binary mask files (one .nii.gz per structure, as
          produced by a non-multilabel TotalSegmentator run), or
        - a single multilabel .nii.gz file (as produced by --ml), where each structure is a
          distinct integer label.

    ct_path: optional path to the CT the masks were derived from, used for intensity_metrics.
        If not given, the intensity_metrics keys are set to None. Must have the same shape as
        each mask (raises ValueError otherwise), since a shape mismatch means the CT and mask
        are not in the same space and any HU statistics computed from them would be meaningless.

    task, class_map: only used for a multilabel file, to map label index -> structure name.
        class_map (e.g. {1: "spleen", ...}) takes precedence if given; otherwise it is looked up
        from the task registry for `task` (default "total").

    Returns: {structure_name: {**volume_metrics, **shape_metrics, **component_metrics,
        **boundary_metrics, **intensity_metrics}}
    """
    mask_path = Path(mask_path)
    ct = None
    if ct_path is not None:
        ct_img = nib.load(ct_path)
        ct = ct_img.get_fdata()

    def metrics_for(mask: np.ndarray, spacing: tuple) -> dict:
        if ct is not None and ct.shape != mask.shape:
            raise ValueError(f"CT shape {ct.shape} does not match mask shape {mask.shape}")
        result = {}
        result.update(volume_metrics(mask, spacing))
        result.update(shape_metrics(mask, spacing))
        result.update(component_metrics(mask))
        result.update(boundary_metrics(mask))
        result.update(intensity_metrics(mask, ct) if ct is not None else _EMPTY_INTENSITY_METRICS)
        return result

    if mask_path.is_dir():
        mask_files = sorted(mask_path.glob("*.nii.gz"))
        metrics = {}
        for mask_file in mask_files:
            img = nib.load(mask_file)
            mask = img.get_fdata() > 0
            structure_name = mask_file.name[:-len(".nii.gz")]
            metrics[structure_name] = metrics_for(mask, img.header.get_zooms())
        return metrics

    img = nib.load(mask_path)
    spacing = img.header.get_zooms()
    data = img.get_fdata()

    if class_map is None:
        from totalsegmentator.registry import get_task_classes
        class_map = get_task_classes(task)

    metrics = {}
    for label_idx, structure_name in class_map.items():
        metrics[structure_name] = metrics_for(data == label_idx, spacing)
    return metrics
