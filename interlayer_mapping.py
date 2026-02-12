import os
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
from scipy.io import savemat


def load_nifti_volume(path: str) -> np.ndarray:
    """Load .nii/.nii.gz as a numpy array."""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)  # float is fine; labels will be cast later
    return data


def resize_nearest_3d(vol: np.ndarray, zoom_factors) -> np.ndarray:
    """
    MATLAB imresizen(...,'nearest')）
    zoom_factors: (zx, zy, zz)
    """
    # order=0 => nearest neighbor
    out = zoom(vol, zoom=zoom_factors, order=0)
    return out


def build_roi_masks(atlas_labels: np.ndarray, roi_num: int, zoom_factors, out_shape=(76, 61, 63)) -> np.ndarray:
    """
    build binary mask for each ROI, resize it with nearest neighbor interpolation
    return shape: (roi_num, 76, 61, 63) uint8 array（0/1）
    """
    masks = np.zeros((roi_num, *out_shape), dtype=np.uint8)

    atlas_int = atlas_labels.astype(np.int32)

    for i in range(1, roi_num + 1):
        roi = (atlas_int == i).astype(np.uint8)  # 0/1
        roi_rs = resize_nearest_3d(roi, zoom_factors).astype(np.uint8)

        ## backup post-processing for python for special cases
        # roi_rs[roi_rs != 0] = 1
        #if roi_rs.shape != out_shape:
        #    fixed = np.zeros(out_shape, dtype=np.uint8)
        #    min_x = min(out_shape[0], roi_rs.shape[0])
        #    min_y = min(out_shape[1], roi_rs.shape[1])
        #    min_z = min(out_shape[2], roi_rs.shape[2])
        #    fixed[:min_x, :min_y, :min_z] = roi_rs[:min_x, :min_y, :min_z]
        #    roi_rs = fixed

        masks[i - 1] = roi_rs

    return masks


def compute_mapping(atlas1_roi: np.ndarray, atlas2_roi: np.ndarray) -> np.ndarray:
    """
    mapping(i,j) = overlapsize / basesize
    overlapsize: number of voxels for boths ROI == 1 
    basesize:  number of voxel for ROI in atlas1
    """
    roi1_num = atlas1_roi.shape[0]
    roi2_num = atlas2_roi.shape[0]
    mapping = np.zeros((roi1_num, roi2_num), dtype=np.float32)

    for i in range(roi1_num):
        base = atlas1_roi[i].sum()
        if base == 0:
            continue
        for j in range(roi2_num):
            overlap = np.logical_and(atlas1_roi[i] == 1, atlas2_roi[j] == 1).sum()
            mapping[i, j] = overlap / float(base)

    return mapping


def main():
    atlas_path = "./atlas"
    output_path = "./interlayermapping"
    os.makedirs(output_path, exist_ok=True)

    atlas1_file = os.path.join(atlas_path, "Schaefer2018_300Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")
    atlas2_file = os.path.join(atlas_path, "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm.nii.gz")

    # === Read atlases ===
    atlas1 = load_nifti_volume(atlas1_file)
    atlas2 = load_nifti_volume(atlas2_file)

    roi_num1 = len(np.unique(atlas1.astype(np.int32))) - 1
    roi_num2 = len(np.unique(atlas2.astype(np.int32))) - 1

    zoom_factors = (76 / 218, 61 / 182, 63 / 182)

    # === Build ROI masks (resized) ===
    atlas1_roi = build_roi_masks(atlas1, roi_num1, zoom_factors, out_shape=(76, 61, 63))
    atlas2_roi = build_roi_masks(atlas2, roi_num2, zoom_factors, out_shape=(76, 61, 63))

    # === Compute overlap mapping ===
    mapping = compute_mapping(atlas1_roi, atlas2_roi)

    # === Save mapping matrices (MATLAB .mat) ===
    out1 = os.path.join(output_path, f"mapping_{roi_num1}to{roi_num2}.mat")
    savemat(out1, {"mapping": mapping})
    
    th = 0 # set any threshold
    mapping_b = (mapping > th).astype(np.uint8)
    out2 = os.path.join(output_path, f"mapping_{roi_num1}to{roi_num2}_b.mat")
    savemat(out2, {"mapping": mapping_b})

    print(f"Saved:\n- {out1}\n- {out2}")
    print(f"mapping shape: {mapping.shape}, roi1={roi_num1}, roi2={roi_num2}")


if __name__ == "__main__":
    main()
