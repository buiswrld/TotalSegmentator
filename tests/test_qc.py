import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

from totalsegmentator.qc import (
    volume_metrics, shape_metrics, component_metrics, boundary_metrics, intensity_metrics,
    calculate_mask_metrics,
)

REFERENCE_DIR = Path(__file__).parent / "reference_files"


# ---- volume_metrics ----

def test_volume_metrics_counts_voxels_and_volume():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True  # 3x3x3 = 27 voxels

    m = volume_metrics(mask, spacing=(2, 2, 2))

    assert m["num_voxels"] == 27
    assert m["volume_mm3"] == round(27 * 8, 2)
    assert m["is_empty"] == 0


def test_volume_metrics_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)

    m = volume_metrics(mask, spacing=(1, 1, 1))

    assert m["num_voxels"] == 0
    assert m["volume_mm3"] == 0.0
    assert m["is_empty"] == 1


# ---- shape_metrics ----

def test_shape_metrics_cube_at_known_position():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True  # indices 2,3,4 -> centroid 3, bbox extent 3

    m = shape_metrics(mask, spacing=(1, 1, 1))

    assert m["centroid_x_rel"] == 0.3
    assert m["centroid_y_rel"] == 0.3
    assert m["centroid_z_rel"] == 0.3
    assert m["bbox_x_rel"] == 0.3
    assert m["bbox_y_rel"] == 0.3
    assert m["bbox_z_rel"] == 0.3
    assert m["bbox_volume_mm3"] == round(27 * 1, 2)
    assert m["mask_to_bbox_ratio"] == 1.0  # solid cube fills its bbox exactly


def test_shape_metrics_sparse_mask_has_low_bbox_ratio():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0, 0, 0] = True
    mask[4, 4, 4] = True  # only 2 voxels spread across a 5x5x5 bbox

    m = shape_metrics(mask, spacing=(1, 1, 1))

    assert m["mask_to_bbox_ratio"] == round(2 / 125, 4)


def test_shape_metrics_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)

    m = shape_metrics(mask, spacing=(1, 1, 1))

    assert m["centroid_x_rel"] is None
    assert m["bbox_volume_mm3"] == 0.0
    assert m["mask_to_bbox_ratio"] == 0.0


# ---- component_metrics ----

def test_component_metrics_single_blob():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:5, 2:5, 2:5] = True

    m = component_metrics(mask)

    assert m["num_components"] == 1
    assert m["largest_component_fraction"] == 1.0


def test_component_metrics_two_disconnected_blobs():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:2, 0:2, 0:2] = True  # 8 voxels
    mask[6:9, 6:9, 6:9] = True  # 27 voxels, disconnected from the first

    m = component_metrics(mask)

    assert m["num_components"] == 2
    assert m["largest_component_fraction"] == round(27 / 35, 4)


def test_component_metrics_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)

    m = component_metrics(mask)

    assert m["num_components"] == 0
    assert m["largest_component_fraction"] == 0.0


# ---- boundary_metrics ----

def test_boundary_metrics_interior_mask_does_not_touch():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[4:7, 4:7, 4:7] = True

    m = boundary_metrics(mask)

    assert m["touches_boundary"] == 0
    assert m["boundary_fraction"] == 0.0


def test_boundary_metrics_mask_touching_one_face():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[0:3, 4:7, 4:7] = True  # touches the x=0 face (within touches_border's 3-voxel check)

    m = boundary_metrics(mask)

    assert m["touches_boundary"] == 1
    assert m["boundary_fraction"] > 0.0


def test_boundary_metrics_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)

    m = boundary_metrics(mask)

    assert m["touches_boundary"] == 0
    assert m["boundary_fraction"] == 0.0


# ---- intensity_metrics ----

def test_intensity_metrics_computes_hu_stats():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[4:7, 4:7, 4:7] = True  # 27 voxels
    ct = np.full((10, 10, 10), -1000.0)
    ct[mask] = 50.0

    m = intensity_metrics(mask, ct)

    assert m["mean_HU"] == 50.0
    assert m["median_HU"] == 50.0
    assert m["std_HU"] == 0.0
    assert m["p05_HU"] == 50.0
    assert m["p95_HU"] == 50.0


def test_intensity_metrics_empty_mask():
    mask = np.zeros((10, 10, 10), dtype=bool)
    ct = np.zeros((10, 10, 10))

    m = intensity_metrics(mask, ct)

    assert m == {"mean_HU": None, "median_HU": None, "std_HU": None, "p05_HU": None, "p95_HU": None}


# ---- calculate_mask_metrics orchestrator ----

def _write_nifti(path, data, spacing=(1.0, 1.0, 1.0)):
    affine = np.diag(list(spacing) + [1.0])
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, path)


def test_calculate_mask_metrics_directory_mode(tmp_path):
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()

    spleen = np.zeros((10, 10, 10), dtype=np.uint8)
    spleen[4:7, 4:7, 4:7] = 1
    _write_nifti(mask_dir / "spleen.nii.gz", spleen)

    kidney = np.zeros((10, 10, 10), dtype=np.uint8)
    _write_nifti(mask_dir / "kidney_right.nii.gz", kidney)  # empty mask

    metrics = calculate_mask_metrics(mask_dir)

    assert metrics["spleen"]["num_voxels"] == 27
    assert metrics["spleen"]["is_empty"] == 0
    assert metrics["kidney_right"]["is_empty"] == 1
    assert metrics["kidney_right"]["mean_HU"] is None  # no ct_path given


def test_calculate_mask_metrics_multilabel_mode(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:7, 4:7, 4:7] = 1
    mask_file = tmp_path / "seg.nii.gz"
    _write_nifti(mask_file, data)

    class_map = {1: "spleen", 2: "liver"}
    metrics = calculate_mask_metrics(mask_file, class_map=class_map)

    assert set(metrics) == {"spleen", "liver"}
    assert metrics["spleen"]["num_voxels"] == 27
    assert metrics["liver"]["is_empty"] == 1


def test_calculate_mask_metrics_with_ct_computes_intensity(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:7, 4:7, 4:7] = 1
    mask_file = tmp_path / "seg.nii.gz"
    _write_nifti(mask_file, data)

    ct = np.full((10, 10, 10), -1000.0)
    ct[4:7, 4:7, 4:7] = 40.0
    ct_file = tmp_path / "ct.nii.gz"
    _write_nifti(ct_file, ct)

    metrics = calculate_mask_metrics(mask_file, ct_path=ct_file, class_map={1: "spleen"})

    assert metrics["spleen"]["mean_HU"] == 40.0


def test_calculate_mask_metrics_ct_shape_mismatch_raises(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.uint8)
    data[4:7, 4:7, 4:7] = 1
    mask_file = tmp_path / "seg.nii.gz"
    _write_nifti(mask_file, data)

    ct_file = tmp_path / "ct.nii.gz"
    _write_nifti(ct_file, np.zeros((12, 12, 12)))  # different shape

    with pytest.raises(ValueError):
        calculate_mask_metrics(mask_file, ct_path=ct_file, class_map={1: "spleen"})


# ---- regression check against the repo's real reference fixtures ----

def test_calculate_mask_metrics_matches_reference_statistics():
    """
    volume_mm3/mean_HU computed by calculate_mask_metrics on the checked-in
    example_seg_fast/ masks + example_ct_sm.nii.gz must agree with
    example_seg_fast/statistics.json, which was produced independently by
    get_basic_statistics (the tool's own runtime stats function) on the same data.
    volume_mm3 uses the same units as statistics.json's "volume" field (mm3), so this is
    a direct comparison with no unit conversion. Agreement here is evidence the
    volume/intensity math in qc.py is correct, not just internally consistent on synthetic
    arrays.
    """
    mask_dir = REFERENCE_DIR / "example_seg_fast"
    ct_path = REFERENCE_DIR / "example_ct_sm.nii.gz"
    with open(mask_dir / "statistics.json") as f:
        ref_stats = json.load(f)

    metrics = calculate_mask_metrics(mask_dir, ct_path=ct_path)

    checked = 0
    for structure_name, ref in ref_stats.items():
        if structure_name not in metrics:
            continue  # statistics.json may include ROIs without a saved mask file
        m = metrics[structure_name]
        if m["is_empty"]:
            # get_basic_statistics uses 0.0 as a "no data" sentinel for empty/border-excluded
            # masks; qc.py deliberately uses None instead, so there's nothing to compare here.
            continue
        assert m["volume_mm3"] == pytest.approx(ref["volume"], abs=1.0)
        assert m["mean_HU"] == pytest.approx(ref["intensity"], abs=0.01)
        checked += 1

    assert checked > 0  # sanity: the fixtures actually overlapped and were compared
