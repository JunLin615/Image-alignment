
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from skimage import color
from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import MiniBatchKMeans
import argparse

def write_results_description(results_root):
    os.makedirs(results_root, exist_ok=True)
    txt_path = os.path.join(results_root, "Results_Description.txt")

    content = r"""结果文件说明
==============================

根目录：
    data/Results_Analysis/

每个 ROI 的结果保存在：
    data/Results_Analysis/ROI_<roi_id>/

文件命名规则：
    ROI_<roi_id>_<编号>_<名称>.<扩展名>

--------------------------------
09_Aligned_RIE_RGB.tiff
--------------------------------
含义：
    对齐后的 RIE RGB 图像（原始分析数据，不做颜色拉伸）。

来源：
    对 moving 原始 RGB 图像应用配准变换后得到。

说明：
    该图用于后续逐像素分析。
    为避免颜色空间被人为改变，这里不做分位数拉伸，也不做逐通道独立拉伸。
    由于重采样/插值，对齐后图像通常保存为 float TIFF，以保留插值后的真实数值。

--------------------------------
10_Original_RGB.tiff
--------------------------------
含义：
    Original 原始 RGB 图像的原样复制。

说明：
    完全保持原始颜色空间、原始通道比例和原始像素值。
    这是 fixed 图像，也就是参考图像。

--------------------------------
11_Raw_RIE_RGB.tiff
--------------------------------
含义：
    RIE 原始 RGB 图像的原样复制（未对齐）。

说明：
    完全保持原始颜色空间、原始通道比例和原始像素值。
    这是 moving 图像，即待对齐图像。

--------------------------------
12_Original_RGB_View.tiff
13_Raw_RIE_RGB_View.tiff
14_Aligned_RIE_RGB_View.tiff
--------------------------------
含义：
    仅用于论文/报告预览的显示版 RGB 图像。

显示规则：
    只允许使用全通道共享的单一线性增益进行亮度压缩到 8-bit，
    不做逐通道独立拉伸，不减去各通道自己的黑电平，
    因此尽量保持 RGB 通道之间的颜色比例不变。

说明：
    这 3 张图仅用于“看图更方便”，不参与后续定量分析。
    其中 14_Aligned_RIE_RGB_View.tiff 使用和 13 相同的显示缩放参数，
    便于前后对比。

--------------------------------
00_Registration_QC.tiff
--------------------------------
含义：
    配准质检图，红色通道来自 Original 灰度，绿色通道来自 Aligned RIE 灰度。

说明：
    偏黄表示对齐较好，偏红/偏绿表示局部偏差。
    这是显示图，不作为定量分析输入。

--------------------------------
01_Diff_Gray.tiff
--------------------------------
含义：
    灰度绝对差图。

计算方式：
    基于原始灰度数据计算：
        |Gray_Original_raw - Gray_AlignedRIE_raw|

说明：
    差图本身由原始数据得到；
    保存为 TIFF 时会单独做显示映射，仅用于可视化。

--------------------------------
02_Diff_RGB_Contrast.tiff
--------------------------------
含义：
    RGB 三通道绝对差图。

计算方式：
    基于原始 RGB 数据逐通道计算：
        Diff_R = |R1 - R2|
        Diff_G = |G1 - G2|
        Diff_B = |B1 - B2|

说明：
    差值计算本身基于原始数据；
    保存图像时会做显示映射，仅用于可视化。

--------------------------------
03_WCAG_Contrast.tiff
--------------------------------
含义：
    WCAG对比度。

计算方式：
    WCAG（Web 内容无障碍指南）标准颜色对比度算法，计算对齐后，RIE前后的颜色对比度。



--------------------------------
04_Ratio_Image.tiff
--------------------------------
含义：
    灰度比值图。

计算方式：
    基于原始灰度数据逐像素计算：
        Gray_Original_raw / (Gray_AlignedRIE_raw + eps)

说明：
    若保存显示时做了范围压缩，则 TIFF 中看到的是显示映射后的结果，
    但比值计算本身始终来自原始数据。

--------------------------------
05_LAB_Euclidean_Distance.tiff
--------------------------------
含义：
    LAB 色彩空间欧氏距离图。

计算方式：
    先将两张原始 RGB 图在“共享单标量归一化”后转换为 LAB，
    再计算每个像素的颜色距离：
        sqrt((L1-L2)^2 + (a1-a2)^2 + (b1-b2)^2)

说明：
    这里没有使用分位数拉伸后的 RGB 图做 LAB，
    以避免不可追溯的颜色扭曲。

--------------------------------
06_SSIM_Structural_Difference.tiff
--------------------------------
含义：
    基于 SSIM 的结构差异图。

计算方式：
    直接对原始灰度数据计算 SSIM map，
    再取：
        1 - SSIM_map

说明：
    不再基于拉伸后的 8-bit 灰度图计算。

--------------------------------
07_2D_Histogram.png
--------------------------------
含义：
    二维灰度直方图（cytofluorogram 风格）。raw 版本基于原始灰度数据，未对齐版本基于原始灰度数据。

计算方式：
    直接使用 Original 原始灰度和 Aligned RIE 原始灰度的对应像素值统计联合分布。

--------------------------------
08_Scatter_Clustering.png
--------------------------------
含义：
    原始灰度散点图及 K-Means 聚类结果。raw 版本基于原始灰度数据，未对齐版本基于原始灰度数据。

计算方式：
    直接使用原始灰度数据作为二维点：
        (Original_gray_raw, RIE_gray_raw)

说明：
    聚类分析不再基于拉伸后的显示图。

--------------------------------
关于“用于显示”和“用于分析”的区别
--------------------------------
本脚本中严格区分两类数据：

1. 原始/分析数据：
   - 10_Original_RGB.tiff
   - 11_Raw_RIE_RGB.tiff
   - 09_Aligned_RIE_RGB.tiff
   - 以及所有定量分析的内部输入

2. 显示增强数据：
   - 12_Original_RGB_View.tiff
   - 13_Raw_RIE_RGB_View.tiff
   - 14_Aligned_RIE_RGB_View.tiff
   - 以及差异图保存时的可视化映射

原则：
    后续分析始终基于原始数据；
    显示增强图仅服务于观看，不参与计算。

补充说明：
    除 09-14 这些原图复制/显示增强结果外，
    00-08 现均会额外保存同名 CSV，
    用于记录对应图像/图表在显示映射前的原始浮点数据。
"""

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

    print(f"[INFO] 结果说明文件已写入: {txt_path}")


def sitk_to_np(img):
    return sitk.GetArrayFromImage(img)


def np_to_sitk(arr, ref_img, is_vector=False):
    out = sitk.GetImageFromArray(arr, isVector=is_vector)
    out.CopyInformation(ref_img)
    return out


def ensure_rgb_float32(img):
    components = img.GetNumberOfComponentsPerPixel()
    
    # 如果是单通道灰度图，把它复制 3 份伪装成 RGB
    if components == 1:
        img = sitk.Compose(img, img, img)
        print("  [提示] 输入图像为单通道灰度图，已自动转为三通道 RGB。")
        
    # 如果是 4 通道 RGBA 图，丢弃 Alpha 透明通道，只保留 RGB
    elif components == 4:
        r = sitk.VectorIndexSelectionCast(img, 0)
        g = sitk.VectorIndexSelectionCast(img, 1)
        b = sitk.VectorIndexSelectionCast(img, 2)
        img = sitk.Compose(r, g, b)
        print("  [提示] 输入图像为 4 通道 RGBA，已自动丢弃 Alpha 通道。")
        
    # 如果既不是 1、3、4，再报错
    elif components != 3:
        raise ValueError(f"Expected 1, 3, or 4 components, but got {components}.")

    return sitk.Cast(img, sitk.sitkVectorFloat32)


def robust_window_np(arr, low_pct=1.0, high_pct=99.0, ignore_zero=False):
    arr = np.asarray(arr, dtype=np.float32)
    valid = np.isfinite(arr)
    if ignore_zero:
        valid &= (arr != 0)

    vals = arr[valid]
    if vals.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)

    lo = np.percentile(vals, low_pct)
    hi = np.percentile(vals, high_pct)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(vals))
        hi = float(np.max(vals))

    if hi <= lo:
        return np.zeros_like(arr, dtype=np.uint8)

    out = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

def cast_rgb_for_save(img_float, ref_native):
    """
    把配准后的 RGB float 图像，转回与原始图一致的 RGB 存储类型，
    这样大多数 TIFF 查看器都会按彩色图显示。

    img_float: 配准后的 RGB 图（通常是 VectorFloat32）
    ref_native: 原始读入的 RGB 图，用来提供目标 dtype
    """
    arr = sitk.GetArrayFromImage(img_float)          # [H, W, 3]
    ref_arr = sitk.GetArrayFromImage(ref_native)     # 用它判断原始 dtype

    # 整数型原图：先四舍五入，再裁剪到合法范围
    if np.issubdtype(ref_arr.dtype, np.integer):
        info = np.iinfo(ref_arr.dtype)
        arr = np.rint(arr)
        arr = np.clip(arr, info.min, info.max).astype(ref_arr.dtype)
    else:
        arr = arr.astype(ref_arr.dtype)

    out = sitk.GetImageFromArray(arr, isVector=True)
    out.CopyInformation(img_float)   # 保留对齐后的空间信息
    return out
def robust_scalar_to_uint8(img, low_pct=1.0, high_pct=99.0, ignore_zero=False):
    arr = sitk_to_np(sitk.Cast(img, sitk.sitkFloat32))
    out = robust_window_np(arr, low_pct, high_pct, ignore_zero)
    return np_to_sitk(out, img, is_vector=False)


def robust_scalar_to_unit_float(img, low_pct=1.0, high_pct=99.0, ignore_zero=False):
    arr = sitk_to_np(sitk.Cast(img, sitk.sitkFloat32))
    arr_u8 = robust_window_np(arr, low_pct, high_pct, ignore_zero)
    arr_f = arr_u8.astype(np.float32) / 255.0
    return np_to_sitk(arr_f, img, is_vector=False)


def save_viewable_tiff(img, path, low_pct=1.0, high_pct=99.0, ignore_zero=False):
    out = robust_scalar_to_uint8(img, low_pct, high_pct, ignore_zero)
    sitk.WriteImage(out, path)


def save_scalar_csv(array_2d, csv_path):
    arr = np.asarray(array_2d, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"save_scalar_csv expects a 2D array, got shape {arr.shape}.")
    np.savetxt(csv_path, arr, delimiter=",", fmt="%.8f")


def save_multichannel_csv(array_hwc, csv_path, channel_names=None):
    arr = np.asarray(array_hwc, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"save_multichannel_csv expects a 3D array [H, W, C], got shape {arr.shape}.")

    h, w, c = arr.shape
    yy, xx = np.indices((h, w))
    flat = np.column_stack((yy.reshape(-1), xx.reshape(-1), arr.reshape(-1, c)))

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(c)]
    if len(channel_names) != c:
        raise ValueError(f"channel_names length {len(channel_names)} does not match channel count {c}.")

    header = "row,col," + ",".join(channel_names)
    fmt = ["%d", "%d"] + ["%.8f"] * c
    np.savetxt(csv_path, flat, delimiter=",", fmt=fmt, header=header, comments="")


def save_point_pairs_csv(x_values, y_values, csv_path, x_name="x", y_name="y", labels=None, label_name="cluster"):
    x_arr = np.asarray(x_values, dtype=np.float32).reshape(-1)
    y_arr = np.asarray(y_values, dtype=np.float32).reshape(-1)
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Point arrays must have the same shape, got {x_arr.shape} vs {y_arr.shape}.")

    cols = [x_arr, y_arr]
    header_parts = [x_name, y_name]
    fmt = ["%.8f", "%.8f"]

    if labels is not None:
        label_arr = np.asarray(labels).reshape(-1)
        if label_arr.shape != x_arr.shape:
            raise ValueError(f"labels shape {label_arr.shape} does not match point shape {x_arr.shape}.")
        cols.append(label_arr)
        header_parts.append(label_name)
        fmt.append("%d")

    data = np.column_stack(cols)
    np.savetxt(
        csv_path,
        data,
        delimiter=",",
        fmt=fmt,
        header=",".join(header_parts),
        comments=""
    )


def rgb_to_luminance_sitk(rgb_img):
    r = sitk.VectorIndexSelectionCast(rgb_img, 0, sitk.sitkFloat32)
    g = sitk.VectorIndexSelectionCast(rgb_img, 1, sitk.sitkFloat32)
    b = sitk.VectorIndexSelectionCast(rgb_img, 2, sitk.sitkFloat32)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return sitk.Cast(gray, sitk.sitkFloat32)


def build_registration_mask(gray_unit):
    mask = sitk.OtsuThreshold(gray_unit, 0, 1, 128)
    mask = sitk.BinaryMorphologicalClosing(mask, [5, 5])
    mask = sitk.BinaryFillhole(mask)
    return sitk.Cast(mask, sitk.sitkUInt8)


def build_initial_transform(fixed_img, moving_img, fixed_mask, moving_mask, use_moments=False):
    if use_moments:
        # 使用基于图像灰度重心的初始化
        print("  [提示] 已启用重心对齐 (Moments) 作为初始位置...")
        t = sitk.CenteredTransformInitializer(
            fixed_img, 
            moving_img, 
            sitk.Euler2DTransform(), 
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        return t
    else:
        # 原来的默认逻辑：物理中心对齐
        size = fixed_img.GetSize()
        center_idx = [(size[0] - 1) / 2.0, (size[1] - 1) / 2.0]
        physical_center = fixed_img.TransformContinuousIndexToPhysicalPoint(center_idx)

        t = sitk.Euler2DTransform()
        t.SetCenter(physical_center)
        t.SetTranslation((0.0, 0.0))
        t.SetAngle(0.0)
        return t


def coarse_rotation_search(
    fixed_img,
    moving_img,
    fixed_mask,
    moving_mask,
    base_transform,
    angle_range_deg=3,
    angle_step_deg=0.1
):
    eval_reg = sitk.ImageRegistrationMethod()
    eval_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    eval_reg.SetMetricSamplingStrategy(eval_reg.REGULAR)
    eval_reg.SetMetricSamplingPercentage(0.5)
    eval_reg.SetInterpolator(sitk.sitkLinear)
    eval_reg.SetMetricFixedMask(fixed_mask)
    eval_reg.SetMetricMovingMask(moving_mask)

    if not hasattr(eval_reg, "MetricEvaluate"):
        return base_transform

    best_metric = np.inf
    best_transform = sitk.Euler2DTransform(base_transform)

    for angle_deg in np.arange(-angle_range_deg, angle_range_deg + 1e-9, angle_step_deg):
        t = sitk.Euler2DTransform(base_transform)
        t.SetAngle(np.deg2rad(float(angle_deg)))
        eval_reg.SetInitialTransform(t, inPlace=False)
        metric_value = eval_reg.MetricEvaluate(fixed_img, moving_img)

        if metric_value < best_metric:
            best_metric = metric_value
            best_transform = sitk.Euler2DTransform(t)

    return best_transform

def save_wcag_contrast_tiff_and_csv(
    rgb1,
    rgb2,
    tiff_path,
    csv_path=None,
    low_pct=1.0,
    high_pct=99.0,
    valid_eps=1e-8
):
    """
    按指定的 WCAG 风格公式计算逐像素对比度，并同时保存：
    1) 原始对比度矩阵 csv
    2) 便于观察的 tiff

    计算规则：
        L = 0.2126*R + 0.7152*G + 0.0722*B
        contrast = (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)

    显示规则：
    - CSV 保存原始 contrast
    - TIFF 不直接对 contrast 做拉伸，而是对 delta = contrast - 1 做显示
    - 先排除“应裁掉/补零”的无效边界，再只用有效区域做分位数统计
    - 无效区域在显示图中补 0（黑色）
    """
    arr1 = sitk_to_np(sitk.Cast(rgb1, sitk.sitkVectorFloat32))
    arr2 = sitk_to_np(sitk.Cast(rgb2, sitk.sitkVectorFloat32))

    if arr1.shape != arr2.shape or arr1.ndim != 3 or arr1.shape[-1] != 3:
        raise ValueError(
            "save_wcag_contrast_tiff_and_csv expects two RGB images with identical shape [H, W, 3]."
        )

    finite1 = arr1[np.isfinite(arr1)]
    finite2 = arr2[np.isfinite(arr2)]

    scale_candidates = []
    if finite1.size > 0:
        scale_candidates.append(float(np.max(finite1)))
    if finite2.size > 0:
        scale_candidates.append(float(np.max(finite2)))

    shared_scale = max(scale_candidates) if scale_candidates else 1.0
    if not np.isfinite(shared_scale) or shared_scale <= 0:
        shared_scale = 1.0

    arr1_unit = np.clip(arr1 / shared_scale, 0.0, 1.0).astype(np.float32)
    arr2_unit = np.clip(arr2 / shared_scale, 0.0, 1.0).astype(np.float32)

    L1 = (
        0.2126 * arr1_unit[..., 0]
        + 0.7152 * arr1_unit[..., 1]
        + 0.0722 * arr1_unit[..., 2]
    )
    L2 = (
        0.2126 * arr2_unit[..., 0]
        + 0.7152 * arr2_unit[..., 1]
        + 0.0722 * arr2_unit[..., 2]
    )

    L_high = np.maximum(L1, L2)
    L_low = np.minimum(L1, L2)

    contrast = ((L_high + 0.05) / (L_low + 0.05)).astype(np.float32)
    contrast[~np.isfinite(contrast)] = 1.0

    if csv_path is None:
        csv_path = os.path.splitext(tiff_path)[0] + ".csv"

    np.savetxt(csv_path, contrast, delimiter=",", fmt="%.8f")

    # 用 delta = contrast - 1 来做显示，更符合“差异热图”的直觉
    delta = contrast - 1.0
    delta[~np.isfinite(delta)] = 0.0

    # 构造有效区域：
    # 只有两张图在该像素都不是“全通道近零”时，才参与统计
    mag1 = np.max(np.abs(arr1_unit), axis=-1)
    mag2 = np.max(np.abs(arr2_unit), axis=-1)
    valid_mask = np.isfinite(delta) & (mag1 > valid_eps) & (mag2 > valid_eps)

    display = np.zeros_like(delta, dtype=np.float32)

    vals = delta[valid_mask]
    if vals.size > 0:
        lo = np.percentile(vals, low_pct)
        hi = np.percentile(vals, high_pct)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))

        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            display[valid_mask] = np.clip((delta[valid_mask] - lo) / (hi - lo), 0.0, 1.0)
        else:
            # 如果有效区域几乎全是同一个值，直接保持黑图，避免假亮
            display[valid_mask] = 0.0

    display_u8 = (display * 255.0).astype(np.uint8)

    ref_scalar = sitk.VectorIndexSelectionCast(rgb1, 0, sitk.sitkFloat32)
    display_img = np_to_sitk(display_u8, ref_scalar, is_vector=False)
    sitk.WriteImage(display_img, tiff_path)

    contrast_img = np_to_sitk(contrast.astype(np.float32), ref_scalar, is_vector=False)
    return contrast_img, csv_path
def compute_ratio_preserving_rgb_reference(img, high_pct=99.0, ignore_zero=True):
    arr = sitk_to_np(sitk.Cast(img, sitk.sitkVectorFloat32))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Expected RGB image with shape [H, W, 3].")

    peak = np.max(arr, axis=-1)
    valid = np.isfinite(peak)
    if ignore_zero:
        valid &= (peak > 0)

    vals = peak[valid]
    if vals.size == 0:
        return 1.0

    ref = np.percentile(vals, high_pct)
    if not np.isfinite(ref) or ref <= 0:
        ref = float(np.max(vals))

    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0

    return float(ref)


def apply_ratio_preserving_rgb_view(img, reference_value):
    arr = sitk_to_np(sitk.Cast(img, sitk.sitkVectorFloat32))
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError("Expected RGB image with shape [H, W, 3].")

    ref = float(reference_value) if reference_value is not None else 1.0
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0

    out = np.clip(arr * (255.0 / ref), 0.0, 255.0).astype(np.uint8)
    return np_to_sitk(out, img, is_vector=True)


def rgb_pair_to_shared_unit_float(img1, img2):
    arr1 = sitk_to_np(sitk.Cast(img1, sitk.sitkVectorFloat32))
    arr2 = sitk_to_np(sitk.Cast(img2, sitk.sitkVectorFloat32))

    finite1 = arr1[np.isfinite(arr1)]
    finite2 = arr2[np.isfinite(arr2)]

    candidates = []
    if finite1.size > 0:
        candidates.append(float(np.max(finite1)))
    if finite2.size > 0:
        candidates.append(float(np.max(finite2)))

    scale = max(candidates) if candidates else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    arr1_unit = np.clip(arr1 / scale, 0.0, 1.0).astype(np.float32)
    arr2_unit = np.clip(arr2 / scale, 0.0, 1.0).astype(np.float32)
    return arr1_unit, arr2_unit


def register_rigid_2d(fixed_gray, moving_gray, fixed_mask, moving_mask, use_moments=False):
    # 将参数传递给初始化函数
    initial_transform = build_initial_transform(fixed_gray, moving_gray, fixed_mask, moving_mask, use_moments)
    initial_transform = coarse_rotation_search(
        fixed_gray,
        moving_gray,
        fixed_mask,
        moving_mask,
        initial_transform,
        angle_range_deg=3,
        angle_step_deg=0.1
    )

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=64)
    reg.SetMetricSamplingStrategy(reg.REGULAR)
    reg.SetMetricSamplingPercentage(0.3)
    reg.SetMetricFixedMask(fixed_mask)
    reg.SetMetricMovingMask(moving_mask)
    reg.SetInterpolator(sitk.sitkLinear)

    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=800,
        gradientMagnitudeTolerance=1e-8
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    reg.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = reg.Execute(fixed_gray, moving_gray)
    return final_transform, reg.GetMetricValue()

def save_joint_distribution_and_clustering(
    img_orig_gray,
    img_cmp_gray,
    base_name,
    roi_id,
    file_tag="",
    label_cmp="RIE"
):
    """
    根据两张灰度图生成：
    - 2D histogram
    - KMeans scatter clustering

    参数：
    img_orig_gray : np.ndarray, Original 原始灰度图
    img_cmp_gray  : np.ndarray, 对比灰度图（可传 aligned 或 raw）
    file_tag      : "" 或 "_raw"
    label_cmp     : 图标题/坐标轴标签里显示的名字，如 "RIE" 或 "Raw RIE"
    """
    if img_orig_gray.shape != img_cmp_gray.shape:
        print(f"[{roi_id}] 跳过 07{file_tag} 和 08{file_tag}：两图尺寸不一致 {img_orig_gray.shape} vs {img_cmp_gray.shape}")
        return

    flat_orig = img_orig_gray.flatten()
    flat_cmp = img_cmp_gray.flatten()

    valid = np.isfinite(flat_orig) & np.isfinite(flat_cmp)
    valid &= ((flat_orig != 0) | (flat_cmp != 0))

    f_orig = flat_orig[valid]
    f_cmp = flat_cmp[valid]

    if len(f_orig) == 0:
        print(f"[{roi_id}] 跳过 07{file_tag} 和 08{file_tag}：有效像素为空。")
        return

    hist_csv_path = f"{base_name}_07{file_tag}_2D_Histogram.csv"
    save_point_pairs_csv(
        f_orig,
        f_cmp,
        hist_csv_path,
        x_name="Original_gray_raw",
        y_name=f"{label_cmp.replace(' ', '_')}_gray_raw"
    )

    xy_min = float(np.min([np.min(f_orig), np.min(f_cmp)]))
    xy_max = float(np.max([np.max(f_orig), np.max(f_cmp)]))
    if not np.isfinite(xy_min):
        xy_min = 0.0
    if not np.isfinite(xy_max) or xy_max <= xy_min:
        xy_max = xy_min + 1.0

    # 07 / 07_raw
    plt.figure(figsize=(8, 6))
    plt.hist2d(f_orig, f_cmp, bins=150, cmap='hot', cmin=1)
    plt.colorbar(label='Pixel Count')
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'w--', alpha=0.5)
    plt.title(f"2D Histogram (Original vs {label_cmp}, raw gray) - ROI {roi_id}")
    plt.xlabel("Original Intensity (raw gray)")
    plt.ylabel(f"{label_cmp} Intensity (raw gray)")
    plt.savefig(f"{base_name}_07{file_tag}_2D_Histogram.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 08 / 08_raw
    data_points = np.column_stack((f_orig, f_cmp))
    kmeans = MiniBatchKMeans(n_clusters=3, random_state=42, n_init=5, batch_size=4096)
    labels = kmeans.fit_predict(data_points)

    scatter_csv_path = f"{base_name}_08{file_tag}_Scatter_Clustering.csv"
    save_point_pairs_csv(
        f_orig,
        f_cmp,
        scatter_csv_path,
        x_name="Original_gray_raw",
        y_name=f"{label_cmp.replace(' ', '_')}_gray_raw",
        labels=labels,
        label_name="cluster_id"
    )

    sample_size = min(50000, len(f_orig))
    sample_indices = np.random.choice(len(f_orig), sample_size, replace=False)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        f_orig[sample_indices],
        f_cmp[sample_indices],
        c=labels[sample_indices],
        cmap='viridis',
        s=1,
        alpha=0.3
    )
    plt.plot([xy_min, xy_max], [xy_min, xy_max], 'k--', alpha=0.5)
    plt.colorbar(scatter, label='Cluster ID')
    plt.title(f"K-Means Scatter Clustering (Original vs {label_cmp}, raw gray) - ROI {roi_id}")
    plt.xlabel("Original Intensity (raw gray)")
    plt.ylabel(f"{label_cmp} Intensity (raw gray)")
    plt.savefig(f"{base_name}_08{file_tag}_Scatter_Clustering.png", dpi=300, bbox_inches='tight')
    plt.close()

def process_and_generate_all(fixed_path, moving_path, out_dir, roi_id, use_moments=False):
    print(f"\n[{roi_id}] 开始处理，建立工作目录...")
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.join(out_dir, f"ROI_{roi_id}")

    fixed_rgb_native = sitk.ReadImage(fixed_path)
    moving_rgb_native = sitk.ReadImage(moving_path)

    fixed_rgb_sitk = ensure_rgb_float32(fixed_rgb_native)
    moving_rgb_sitk = ensure_rgb_float32(moving_rgb_native)

    # 原始 RGB 精确复制保存
    sitk.WriteImage(fixed_rgb_native, f"{base_name}_10_Original_RGB.tiff")
    sitk.WriteImage(moving_rgb_native, f"{base_name}_11_Raw_RIE_RGB.tiff")

    # 仅用于显示的 RGB 视图：使用共享单一增益，不改变 RGB 比例
    orig_view_ref = compute_ratio_preserving_rgb_reference(
        fixed_rgb_sitk,
        high_pct=99.0,
        ignore_zero=True
    )
    rie_view_ref = compute_ratio_preserving_rgb_reference(
        moving_rgb_sitk,
        high_pct=99.0,
        ignore_zero=True
    )

    orig_view = apply_ratio_preserving_rgb_view(fixed_rgb_sitk, orig_view_ref)
    sitk.WriteImage(orig_view, f"{base_name}_12_Original_RGB_View.tiff")

    raw_rie_view = apply_ratio_preserving_rgb_view(moving_rgb_sitk, rie_view_ref)
    sitk.WriteImage(raw_rie_view, f"{base_name}_13_Raw_RIE_RGB_View.tiff")

    # 更稳定的灰度：仅用于配准预处理，不改变后续分析使用的原始数据
    fixed_gray_raw = rgb_to_luminance_sitk(fixed_rgb_sitk)
    moving_gray_raw = rgb_to_luminance_sitk(moving_rgb_sitk)

    fixed_gray_reg = robust_scalar_to_unit_float(fixed_gray_raw, low_pct=1, high_pct=99, ignore_zero=True)
    moving_gray_reg = robust_scalar_to_unit_float(moving_gray_raw, low_pct=1, high_pct=99, ignore_zero=True)

    fixed_gray_reg = sitk.DiscreteGaussian(fixed_gray_reg, variance=1.0)
    moving_gray_reg = sitk.DiscreteGaussian(moving_gray_reg, variance=1.0)

    fixed_mask = build_registration_mask(fixed_gray_reg)
    moving_mask = build_registration_mask(moving_gray_reg)

    print(f"[{roi_id}] 正在执行更稳健的刚性对齐...")
    final_transform, final_metric = register_rigid_2d(
        fixed_gray_reg,
        moving_gray_reg,
        fixed_mask,
        moving_mask,
        use_moments=use_moments,
    )
    print(f"[{roi_id}] 配准完成，final metric = {final_metric:.6f}")
    print(f"[{roi_id}] final transform = {final_transform}")

    aligned_moving_rgb_sitk = sitk.Resample(
        moving_rgb_sitk,
        fixed_rgb_sitk,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving_rgb_sitk.GetPixelID()
    )
    aligned_moving_gray_raw = rgb_to_luminance_sitk(aligned_moving_rgb_sitk)

    # 对齐后的原始分析图（不做颜色拉伸）
    aligned_moving_rgb_save = cast_rgb_for_save(aligned_moving_rgb_sitk, moving_rgb_native)
    sitk.WriteImage(aligned_moving_rgb_save, f"{base_name}_09_Aligned_RIE_RGB.tiff")

    # 对齐后的显示图：沿用 moving 的同一套共享增益
    aligned_rie_view = apply_ratio_preserving_rgb_view(aligned_moving_rgb_sitk, rie_view_ref)
    sitk.WriteImage(aligned_rie_view, f"{base_name}_14_Aligned_RIE_RGB_View.tiff")

    # 配准质检图
    qc_r_raw = sitk_to_np(sitk.Cast(fixed_gray_raw, sitk.sitkFloat32)).astype(np.float32)
    qc_g_raw = sitk_to_np(sitk.Cast(aligned_moving_gray_raw, sitk.sitkFloat32)).astype(np.float32)
    qc_b_raw = np.zeros_like(qc_r_raw, dtype=np.float32)
    qc_overlay_raw = np.stack((qc_r_raw, qc_g_raw, qc_b_raw), axis=-1)
    save_multichannel_csv(
        qc_overlay_raw,
        f"{base_name}_00_Registration_QC.csv",
        channel_names=["Original_gray_raw", "Aligned_RIE_gray_raw", "Blue_zero"]
    )

    qc_r = robust_scalar_to_uint8(fixed_gray_raw, ignore_zero=True)
    qc_g = robust_scalar_to_uint8(aligned_moving_gray_raw, ignore_zero=True)
    qc_b = sitk.Image(qc_r.GetSize(), sitk.sitkUInt8) + 0
    qc_b.CopyInformation(qc_r)
    qc_overlay = sitk.Compose(qc_r, qc_g, qc_b)
    sitk.WriteImage(qc_overlay, f"{base_name}_00_Registration_QC.tiff")

    print(f"[{roi_id}] 正在计算基础差异特征 (1-4)...")

    diff_gray = sitk.Abs(fixed_gray_raw - aligned_moving_gray_raw)
    diff_gray_array = sitk_to_np(diff_gray).astype(np.float32)
    save_scalar_csv(diff_gray_array, f"{base_name}_01_Diff_Gray.csv")
    save_viewable_tiff(diff_gray, f"{base_name}_01_Diff_Gray.tiff", ignore_zero=False)

    diff_channels = []
    diff_channel_arrays = []
    for i in range(3):
        fixed_c = sitk.VectorIndexSelectionCast(fixed_rgb_sitk, i, sitk.sitkFloat32)
        moving_c = sitk.VectorIndexSelectionCast(aligned_moving_rgb_sitk, i, sitk.sitkFloat32)
        diff_c = sitk.Abs(fixed_c - moving_c)
        diff_channel_arrays.append(sitk_to_np(diff_c).astype(np.float32))
        diff_channels.append(robust_scalar_to_uint8(diff_c))

    diff_rgb_raw = np.stack(diff_channel_arrays, axis=-1)
    save_multichannel_csv(
        diff_rgb_raw,
        f"{base_name}_02_Diff_RGB_Contrast.csv",
        channel_names=["Diff_R", "Diff_G", "Diff_B"]
    )

    diff_rgb = sitk.Compose(diff_channels[0], diff_channels[1], diff_channels[2])
    sitk.WriteImage(diff_rgb, f"{base_name}_02_Diff_RGB_Contrast.tiff")
    save_wcag_contrast_tiff_and_csv(
        fixed_rgb_sitk,
        aligned_moving_rgb_sitk,
        f"{base_name}_03_WCAG_Contrast.tiff"
    )

    ratio_image = sitk.Divide(fixed_gray_raw + 1e-5, aligned_moving_gray_raw + 1e-5)
    ratio_image = sitk.Clamp(ratio_image, sitk.sitkFloat32, 0.0, 2.0)
    ratio_array = sitk_to_np(ratio_image).astype(np.float32)
    save_scalar_csv(ratio_array, f"{base_name}_04_Ratio_Image.csv")
    save_viewable_tiff(ratio_image, f"{base_name}_04_Ratio_Image.tiff")

    print(f"[{roi_id}] 正在计算高级结构特征与聚类 (5-8)...")

    # LAB：基于原始 RGB，仅做共享单标量归一化，不再使用拉伸后的显示图
    img_orig_rgb_unit, img_rie_rgb_unit = rgb_pair_to_shared_unit_float(
        fixed_rgb_sitk,
        aligned_moving_rgb_sitk
    )
    lab_orig = color.rgb2lab(img_orig_rgb_unit)
    lab_rie = color.rgb2lab(img_rie_rgb_unit)
    lab_distance = np.sqrt(np.sum((lab_orig - lab_rie) ** 2, axis=-1)).astype(np.float32)
    save_scalar_csv(lab_distance, f"{base_name}_05_LAB_Euclidean_Distance.csv")
    lab_sitk = np_to_sitk(lab_distance, fixed_gray_raw, is_vector=False)
    save_viewable_tiff(lab_sitk, f"{base_name}_05_LAB_Euclidean_Distance.tiff")

    # SSIM：直接基于原始灰度数据
    img_orig_gray_raw = sitk_to_np(fixed_gray_raw).astype(np.float32)
    img_rie_gray_raw = sitk_to_np(aligned_moving_gray_raw).astype(np.float32)
    img_rie_gray_raw_unaligned = sitk_to_np(moving_gray_raw).astype(np.float32)
    
    joint_min = float(np.min([np.min(img_orig_gray_raw), np.min(img_rie_gray_raw)]))
    joint_max = float(np.max([np.max(img_orig_gray_raw), np.max(img_rie_gray_raw)]))
    data_range = joint_max - joint_min
    if not np.isfinite(data_range) or data_range <= 0:
        data_range = 1.0

    score, ssim_map = ssim(
        img_orig_gray_raw,
        img_rie_gray_raw,
        full=True,
        data_range=data_range
    )
    ssim_diff = (1.0 - ssim_map).astype(np.float32)
    save_scalar_csv(ssim_diff, f"{base_name}_06_SSIM_Structural_Difference.csv")
    ssim_sitk = np_to_sitk(ssim_diff, fixed_gray_raw, is_vector=False)
    save_viewable_tiff(ssim_sitk, f"{base_name}_06_SSIM_Structural_Difference.tiff")
    print(f"[{roi_id}] SSIM score = {score:.6f}")

    save_joint_distribution_and_clustering(
        img_orig_gray_raw,
        img_rie_gray_raw,
        base_name,
        roi_id,
        file_tag="",
        label_cmp="RIE"
    )

    save_joint_distribution_and_clustering(
        img_orig_gray_raw,
        img_rie_gray_raw_unaligned,
        base_name,
        roi_id,
        file_tag="_raw",
        label_cmp="Raw RIE"
    )

    print(f"[{roi_id}] 所有输出生成完毕！")


def main(data_dir, use_moments=False):  # 增加参数
    files = os.listdir(data_dir)
    pattern = re.compile(r"ROI (\d+) (Original|RIE)\.tiff?")
    results_root = os.path.join(data_dir, "Results_Analysis")
    os.makedirs(results_root, exist_ok=True)
    write_results_description(results_root)

    pairs = {}
    for f in files:
        match = pattern.match(f)
        if match:
            roi_id, role = match.group(1), match.group(2)
            if roi_id not in pairs:
                pairs[roi_id] = {}
            pairs[roi_id][role] = os.path.join(data_dir, f)

    for roi_id, paths in pairs.items():
        if "Original" in paths and "RIE" in paths:
            roi_out_dir = os.path.join(results_root, f"ROI_{roi_id}")
            process_and_generate_all(paths["Original"], paths["RIE"], roi_out_dir, roi_id, use_moments)


if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="处理 ROI 图像的配准与差异分析")
    
    # 添加一个路径参数。
    # 这里设置了一个默认值（你原来的路径），如果命令行不传参，就会默认使用它。
    parser.add_argument(
        "-d", "--data_dir", 
        type=str, 
        default="F:/Jiajun task 20260420", 
        help="指定包含 ROI 图像的根目录路径"
    )
    
    # 新增：添加重心对齐开关 (触发时为 True，不写时默认 False)
    parser.add_argument(
        "--use_moments", 
        action="store_true", 
        help="如果图像初始平移偏差较大，使用此参数启用基于重心的初始对齐"
    )
    # 解析命令行输入的参数
    args = parser.parse_args()
    
    # 将解析到的路径传给 main 函数
    main(args.data_dir, args.use_moments)