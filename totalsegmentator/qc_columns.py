"""
Shared column-name constants for the ground-truth-free QC classifier pipeline.

Pipeline: mask_metrics.py -> evaluate_ct.py / combined.py -> build_reference.py ->
train_classifier.py -> calibrate.py.

Every CSV column that flows through more than one of these files (e.g. combined_metrics.csv
is written by evaluate_ct.py and read by both train_classifier.py and calibrate.py) is
defined here exactly once, as its full descriptive label with the short form in
parentheses, and imported everywhere that column is produced or consumed. This is NOT a
translation/mapping layer - the constant's value IS the real column name used on disk and
in memory throughout the pipeline; defining it once just guarantees the writer and every
reader use the identical literal string instead of retyping it and risking a mismatch.

Whenever a new feature/metric column is added anywhere in this pipeline: add its constant
here, and add a row for it in the "Feature & Metric Glossary" table in RESEARCH.md (see
CLAUDE.md for the standing instruction).
"""

# ---------------------------------------------------------------- identifiers
COL_SUBJECT = "Subject ID (subject)"
COL_ORGAN = "Organ/Structure Name (organ)"
COL_ORIG_AXCODES = "Original Image Axis Orientation Codes (orig_axcodes)"

# ---------------------------------------------------------------- mask_metrics.py
# Ground-truth-free features computed directly from a mask (+ CT/MR volume for
# intensity_metrics). These are the raw features available to the classifier.
COL_NUM_VOXELS = "Number of Voxels (num_voxels)"
COL_VOLUME_MM3 = "Volume in Cubic Millimeters (volume_mm3)"
COL_IS_EMPTY = "Is Mask Empty (is_empty)"
COL_CENTROID_X_REL = "Relative Centroid X Position (centroid_x_rel)"
COL_CENTROID_Y_REL = "Relative Centroid Y Position (centroid_y_rel)"
COL_CENTROID_Z_REL = "Relative Centroid Z Position (centroid_z_rel)"
COL_BBOX_X_REL = "Relative Bounding Box Width (bbox_x_rel)"
COL_BBOX_Y_REL = "Relative Bounding Box Height (bbox_y_rel)"
COL_BBOX_Z_REL = "Relative Bounding Box Depth (bbox_z_rel)"
COL_BBOX_VOLUME_MM3 = "Bounding Box Volume in Cubic Millimeters (bbox_volume_mm3)"
COL_MASK_TO_BBOX_RATIO = "Mask-to-Bounding-Box Fill Ratio (mask_to_bbox_ratio)"
COL_NUM_COMPONENTS = "Number of Connected Components (num_components)"
COL_LARGEST_COMPONENT_FRACTION = "Largest Connected Component Fraction (largest_component_fraction)"
COL_TOUCHES_BOUNDARY = "Touches Image Boundary (touches_boundary)"
COL_BOUNDARY_FRACTION = "Boundary Voxel Fraction (boundary_fraction)"
COL_MEAN_HU = "Mean Hounsfield Unit Intensity (mean_HU)"
COL_MEDIAN_HU = "Median Hounsfield Unit Intensity (median_HU)"
COL_STD_HU = "Hounsfield Unit Intensity Standard Deviation (std_HU)"
COL_P05_HU = "5th Percentile Hounsfield Unit Intensity (p05_HU)"
COL_P95_HU = "95th Percentile Hounsfield Unit Intensity (p95_HU)"

# ---------------------------------------------------------------- evaluate_ct.py / combined.py
# Ground-truth-derived accuracy labels. These require the ground-truth mask, so they are
# used only to build the accept/reject label and must never be fed to the classifier as
# a feature (see LEAK_COLS in train_classifier.py / calibrate.py).
COL_DICE = "Sorensen-Dice Coefficient (Dice)"
COL_IOU = "Intersection over Union (IoU)"
COL_TRUE_POSITIVES = "True Positive Voxel Count (TP)"
COL_FALSE_POSITIVES = "False Positive Voxel Count (FP)"
COL_FALSE_NEGATIVES = "False Negative Voxel Count (FN)"
COL_PREDICTED_VOXEL_COUNT = "Predicted Mask Voxel Count (pred_vox)"
COL_GROUND_TRUTH_VOXEL_COUNT = "Ground Truth Mask Voxel Count (gt_vox)"
COL_MATCH_STATUS = "Match Status (status)"
COL_ACCEPT_LABEL = "Accept/Reject Label (accept)"
COL_TRAINING_LABEL = "Accept/Reject Training Label (label)"

# ---------------------------------------------------------------- build_reference.py
COL_FEATURE_NAME = "Feature/Metric Name (feature)"
COL_OBSERVATION_COUNT = "Number of Reference Observations (n)"
COL_REFERENCE_MEDIAN = "Reference Median Value (median)"
COL_REFERENCE_Q25 = "Reference 25th Percentile (q25)"
COL_REFERENCE_Q75 = "Reference 75th Percentile (q75)"
COL_REFERENCE_IQR = "Reference Interquartile Range (iqr)"
COL_LOG_SPACE_FLAG = "Computed in Log Space (log_space)"

# ---------------------------------------------------------------- left/right audit (evaluate_ct.py)
COL_MASK_SOURCE = "Mask Source: Prediction or Ground Truth (source)"
COL_PAIRED_STRUCTURE = "Paired Structure Base Name (structure)"
COL_LEFT_CENTROID_X_REL = "Left Structure Relative Centroid X (left_centroid_x_rel)"
COL_RIGHT_CENTROID_X_REL = "Right Structure Relative Centroid X (right_centroid_x_rel)"
COL_RIGHT_MINUS_LEFT = "Right Minus Left Centroid X Difference (right_minus_left)"
COL_LR_VERDICT = "Left/Right Orientation Verdict (verdict)"

# ---------------------------------------------------------------- summary/rollup tables
COL_N_PRESENT_BOTH = "Number Present in Both Prediction and Ground Truth (n_present_both)"
COL_N_SCORED = "Number of Masks Scored (n_scored)"
COL_MEAN_DICE = "Mean Sorensen-Dice Coefficient (mean_dice)"
COL_MEAN_IOU = "Mean Intersection over Union (mean_iou)"
COL_N_OK = "Number Passing Dice Threshold (n_ok)"
COL_N_LOW_DICE = "Number Below Dice Threshold (n_low_dice)"
COL_N_MISS = "Number Missed by Prediction (n_miss)"
COL_N_FALSE_POSITIVE = "Number of False Positive Predictions (n_false_positive)"
COL_N_ACCEPT = "Number Accepted (n_accept)"
COL_ACCEPT_RATE = "Accept Rate (accept_rate)"
COL_MEASURED_DICE = "Measured Mean Dice (measured_dice)"
COL_EXPECTED_DICE = "Published Expected Dice (expected_dice)"
COL_DICE_DELTA = "Measured Minus Expected Dice (delta)"
COL_N_SUBJECTS = "Number of Subjects (n_subjects)"
COL_N_MASKS = "Number of Masks (n)"
COL_MEAN_IOU_ACCEPTED = "Mean IoU of Accepted Masks (mean_iou_accepted)"
COL_MEAN_IOU_REJECTED = "Mean IoU of Rejected Masks (mean_iou_rejected)"

# ---------------------------------------------------------------- calibrate.py
COL_MODEL_NAME = "Model Name (model)"
COL_CALIBRATION_METHOD = "Calibration Method (calibration)"
COL_AUC = "Area Under ROC Curve (AUC)"
COL_ECE = "Expected Calibration Error (ECE)"
COL_MCE = "Maximum Calibration Error (MCE)"
COL_BRIER_SCORE = "Brier Score (brier)"
COL_RELIABILITY = "Brier Reliability Component (reliability)"
COL_RESOLUTION = "Brier Resolution Component (resolution)"
COL_UNCERTAINTY = "Brier Uncertainty Component (uncertainty)"
COL_BIN_MEAN_PREDICTED = "Bin Mean Predicted Probability (mean_pred)"
COL_BIN_OBSERVED_RATE = "Bin Observed Accept Rate (observed)"
COL_CALIBRATION_GAP = "Observed Minus Predicted Gap (gap)"

# ---------------------------------------------------------------- train.py / test.py result tables
COL_ROC_AUC_MEAN = "Mean ROC-AUC Across CV Folds (roc_auc_mean)"
COL_ROC_AUC_STD = "ROC-AUC Standard Deviation Across CV Folds (roc_auc_std)"
COL_PR_AUC_MEAN = "Mean PR-AUC Across CV Folds (pr_auc_mean)"
COL_PR_AUC_STD = "PR-AUC Standard Deviation Across CV Folds (pr_auc_std)"
COL_PR_AUC_ACCEPT = "PR-AUC for Accept Class (pr_auc_accept)"
COL_PR_AUC_REJECT = "PR-AUC for Reject Class (pr_auc_reject)"
COL_N_TEST_ROWS = "Number of Test Rows (n_test_rows)"

# ---------------------------------------------------------------- ablations.py
COL_ANATOMICAL_FAMILY = "Anatomical Family Grouping (family)"
COL_N_ORGANS = "Number of Distinct Organs (n_organs)"
COL_ABLATION_CONFIG = "Ablation Configuration Label (config)"
COL_ABLATION_KIND = "Ablation Configuration Kind (kind)"
COL_N_FEATURES = "Number of Features Used (n_features)"
COL_WITHIN_ORGAN_AUC = "Mean Within-Group AUC (within_organ_auc)"
COL_N_GROUPS_WITHIN = "Number of Groups in Within-Group AUC (n_organs_within)"
COL_DELTA_AUC = "Pooled AUC Change vs Full Model (delta_auc)"
COL_DELTA_WITHIN_AUC = "Within-Group AUC Change vs Full Model (delta_within)"
COL_IOU_THRESHOLD = "IoU Accept Threshold Tested (iou_threshold)"
