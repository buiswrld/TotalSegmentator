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
from totalsegmentator.alignment import as_closest_canonical
from totalsegmentator.qc_columns import (
    COL_NUM_VOXELS, COL_VOLUME_MM3, COL_IS_EMPTY,
    COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL,
    COL_BBOX_X_REL, COL_BBOX_Y_REL, COL_BBOX_Z_REL,
    COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO,
    COL_NUM_COMPONENTS, COL_LARGEST_COMPONENT_FRACTION,
    COL_TOUCHES_BOUNDARY, COL_BOUNDARY_FRACTION,
    COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU,
)

CANONICAL_AXCODES = ["R", "A", "S"]


def volume_metrics(mask: np.ndarray, spacing: tuple) -> dict:
    """
    Number of Voxels, Volume in Cubic Millimeters and Is Mask Empty for a binary mask.

    spacing: (x, y, z) voxel spacing in mm, as returned by nib header.get_zooms().

    Volume in Cubic Millimeters is in mm3, matching the units of the "volume" field
    statistics.py writes to statistics.json (see get_basic_statistics), so the two can be
    compared/cross-checked directly without a unit conversion.

    Returns: {COL_NUM_VOXELS: int, COL_VOLUME_MM3: float, COL_IS_EMPTY: int}
    """
    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    num_voxels = int(mask.sum())
    return {
        COL_NUM_VOXELS: num_voxels,
        COL_VOLUME_MM3: round(float(num_voxels * voxel_volume_mm3), 2),
        COL_IS_EMPTY: int(num_voxels == 0),
    }


def shape_metrics(mask: np.ndarray, spacing: tuple) -> dict:
    """
    Relative centroid/bbox position and mask-to-bbox compactness for a binary mask.

    Centroid and bbox extents are expressed as fractions of the image size along each axis,
    so they are comparable across images of different dimensions. bbox extents use an
    inclusive-max convention (max_idx - min_idx + 1).

    centroid_x/y/z_rel and bbox_x/y/z_rel are only comparable across different scans if mask's
    array axes consistently mean the same anatomical direction in every call - NIfTI files can
    store the same anatomy along differently ordered/flipped axes (see nib.aff2axcodes), so
    callers should reorient to a fixed orientation (e.g. via
    totalsegmentator.alignment.as_closest_canonical) before calling this, as
    calculate_mask_metrics does.

    spacing: (x, y, z) voxel spacing in mm.

    bbox_volume_mm3 is in mm3, matching statistics.json's "volume" units (see volume_metrics).

    Returns: {COL_CENTROID_X_REL, COL_CENTROID_Y_REL, COL_CENTROID_Z_REL, COL_BBOX_X_REL,
        COL_BBOX_Y_REL, COL_BBOX_Z_REL, COL_BBOX_VOLUME_MM3, COL_MASK_TO_BBOX_RATIO}
    """
    shape = mask.shape
    voxel_indices = np.argwhere(mask)
    if voxel_indices.shape[0] == 0:
        return {
            COL_CENTROID_X_REL: None, COL_CENTROID_Y_REL: None, COL_CENTROID_Z_REL: None,
            COL_BBOX_X_REL: 0.0, COL_BBOX_Y_REL: 0.0, COL_BBOX_Z_REL: 0.0,
            COL_BBOX_VOLUME_MM3: 0.0, COL_MASK_TO_BBOX_RATIO: 0.0,
        }

    centroid = voxel_indices.mean(axis=0)
    bbox_mins = voxel_indices.min(axis=0)
    bbox_maxs = voxel_indices.max(axis=0)
    bbox_extent_vox = bbox_maxs - bbox_mins + 1
    bbox_extent_rel = bbox_extent_vox / np.array(shape)

    voxel_volume_mm3 = spacing[0] * spacing[1] * spacing[2]
    bbox_voxel_count = int(np.prod(bbox_extent_vox))
    num_voxels = int(voxel_indices.shape[0])

    return {
        COL_CENTROID_X_REL: round(float(centroid[0] / shape[0]), 4),
        COL_CENTROID_Y_REL: round(float(centroid[1] / shape[1]), 4),
        COL_CENTROID_Z_REL: round(float(centroid[2] / shape[2]), 4),
        COL_BBOX_X_REL: round(float(bbox_extent_rel[0]), 4),
        COL_BBOX_Y_REL: round(float(bbox_extent_rel[1]), 4),
        COL_BBOX_Z_REL: round(float(bbox_extent_rel[2]), 4),
        COL_BBOX_VOLUME_MM3: round(float(bbox_voxel_count * voxel_volume_mm3), 2),
        COL_MASK_TO_BBOX_RATIO: round(float(num_voxels / bbox_voxel_count), 4),
    }


def component_metrics(mask: np.ndarray) -> dict:
    """
    Connected-component structure of a binary mask, via scipy.ndimage.label.

    Returns: {COL_NUM_COMPONENTS: int, COL_LARGEST_COMPONENT_FRACTION: float}
    """
    num_voxels = int(mask.sum())
    if num_voxels == 0:
        return {COL_NUM_COMPONENTS: 0, COL_LARGEST_COMPONENT_FRACTION: 0.0}

    labeled, num_components = ndimage.label(mask)
    component_voxel_counts = np.bincount(labeled.flatten())[1:]  # exclude background (label 0)
    largest_component_fraction = float(component_voxel_counts.max() / num_voxels)

    return {
        COL_NUM_COMPONENTS: int(num_components),
        COL_LARGEST_COMPONENT_FRACTION: round(largest_component_fraction, 4),
    }


def boundary_metrics(mask: np.ndarray) -> dict:
    """
    Whether/how much a binary mask touches the edge of the image volume.

    touches_boundary reuses statistics.touches_border, which only inspects the outer 3
    voxels of each face. boundary_fraction is a finer-grained companion metric: the exact
    fraction of the mask's voxels that sit on any of the 6 outermost faces (single-voxel
    layer), which touches_boundary alone cannot distinguish (e.g. one stray voxel vs. an
    entire face).

    Returns: {COL_TOUCHES_BOUNDARY: int, COL_BOUNDARY_FRACTION: float}
    """
    num_voxels = int(mask.sum())
    if num_voxels == 0:
        return {COL_TOUCHES_BOUNDARY: 0, COL_BOUNDARY_FRACTION: 0.0}

    touches = int(touches_border(mask))

    boundary_shell = np.zeros_like(mask, dtype=bool)
    boundary_shell[0, :, :] = True
    boundary_shell[-1, :, :] = True
    boundary_shell[:, 0, :] = True
    boundary_shell[:, -1, :] = True
    boundary_shell[:, :, 0] = True
    boundary_shell[:, :, -1] = True
    boundary_voxels = int(np.count_nonzero(mask & boundary_shell))

    return {
        COL_TOUCHES_BOUNDARY: touches,
        COL_BOUNDARY_FRACTION: round(float(boundary_voxels / num_voxels), 4),
    }


def intensity_metrics(mask: np.ndarray, ct: np.ndarray) -> dict:
    """
    HU intensity distribution of a CT image restricted to a binary mask.

    Returns: {COL_MEAN_HU, COL_MEDIAN_HU, COL_STD_HU, COL_P05_HU, COL_P95_HU}
    """
    if mask.sum() == 0:
        return {COL_MEAN_HU: None, COL_MEDIAN_HU: None, COL_STD_HU: None, COL_P05_HU: None, COL_P95_HU: None}

    masked_intensities = ct[mask]
    return {
        COL_MEAN_HU: round(float(np.mean(masked_intensities)), 2),
        COL_MEDIAN_HU: round(float(np.median(masked_intensities)), 2),
        COL_STD_HU: round(float(np.std(masked_intensities)), 2),
        COL_P05_HU: round(float(np.percentile(masked_intensities, 5)), 2),
        COL_P95_HU: round(float(np.percentile(masked_intensities, 95)), 2),
    }


_EMPTY_INTENSITY_METRICS = {COL_MEAN_HU: None, COL_MEDIAN_HU: None, COL_STD_HU: None, COL_P05_HU: None, COL_P95_HU: None}


def _reorient_to_canonical(img: nib.Nifti1Image):
    """Return (canonical_img, original_axcodes). See calculate_mask_metrics for why."""
    original_axcodes = list(nib.aff2axcodes(img.affine))
    return as_closest_canonical(img), original_axcodes


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
        each mask after reorientation (raises ValueError otherwise), since a shape mismatch
        means the CT and mask are not in the same space and any HU statistics computed from
        them would be meaningless.

    task, class_map: only used for a multilabel file, to map label index -> structure name.
        class_map (e.g. {1: "spleen", ...}) takes precedence if given; otherwise it is looked up
        from the task registry for `task` (default "total").

    Every mask (and the CT, if given) is reoriented to the closest canonical RAS orientation
    (totalsegmentator.alignment.as_closest_canonical) before shape_metrics is computed. NIfTI
    files can store the same anatomy with array axes in any order/flip depending on the
    scanner/conversion pipeline (see nib.aff2axcodes) - without this, centroid_x/y/z_rel and
    bbox_x/y/z_rel from two different scans would not be comparable even though the field
    names look identical. The orientation actually detected (before reorientation) is reported
    under "orientation" so this correction is auditable rather than an invisible assumption.

    Returns: {"structures": {structure_name: {**volume_metrics, **shape_metrics,
        **component_metrics, **boundary_metrics, **intensity_metrics}},
        "orientation": {"original_axcodes": [...], "canonical_axcodes": ["R", "A", "S"]}}
    """
    mask_path = Path(mask_path)
    ct = None
    original_axcodes = None
    if ct_path is not None:
        ct_img, original_axcodes = _reorient_to_canonical(nib.load(ct_path))
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
        structures = {}
        for mask_file in mask_files:
            img, mask_axcodes = _reorient_to_canonical(nib.load(mask_file))
            if original_axcodes is None:
                original_axcodes = mask_axcodes
            mask = img.get_fdata() > 0
            structure_name = mask_file.name[:-len(".nii.gz")]
            structures[structure_name] = metrics_for(mask, img.header.get_zooms())
        return {"structures": structures,
                "orientation": {"original_axcodes": original_axcodes, "canonical_axcodes": list(CANONICAL_AXCODES)}}

    img, mask_axcodes = _reorient_to_canonical(nib.load(mask_path))
    if original_axcodes is None:
        original_axcodes = mask_axcodes
    spacing = img.header.get_zooms()
    data = img.get_fdata()

    if class_map is None:
        from totalsegmentator.registry import get_task_classes
        class_map = get_task_classes(task)

    structures = {}
    for label_idx, structure_name in class_map.items():
        structures[structure_name] = metrics_for(data == label_idx, spacing)
    return {"structures": structures,
            "orientation": {"original_axcodes": original_axcodes, "canonical_axcodes": list(CANONICAL_AXCODES)}}
