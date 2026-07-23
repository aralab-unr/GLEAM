import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import json
import hashlib
import datetime
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import warnings

# Suppress harmless matplotlib layout warnings caused by inset axes
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/splat_data"         ## Directory containing the splat data
JSON_FILE = os.path.join(DATA_DIR, "transforms.json")
PCD_FILE = "/home/ara/ros2_ws/src/robust_lidar_inertial/log/global_map.pcd"

CROSSHAIR_PIXEL = (501, 564) # HARDCODED TARGET ROI         ## Pixel coordinates of the target region of interest
# CROSSHAIR_PIXEL = (897, 299) # HARDCODED TARGET ROI
TARGET_FREQ_HZ = 0.65
FPS = 20.0
DOWNSCALE_FACTOR = 0.25
ROI_BOX_SIZE = 250

# Reverted to match ROI_BOX_SIZE.
FIGURE5_ROI_BOX_SIZE = 250

RANSAC_REPROJ_THRESHOLD = 3.0
RANSAC_MIN_INLIERS = 15
NAIVE_PLANE_ASSUMED_DEPTH = 3.0  # meters; a guessed standoff distance, NOT measured by LiDAR

CLASSIC_STAB_SMOOTHING_WINDOW = 30  # frames (~1.5s at 20 FPS)
OUTLIER_SCALE_FACTOR = 3.0
SCENE_EXTENT_MARGIN_M = 2.0  # meters padded onto each side of the LiDAR map's bounding box
MAX_PLAUSIBLE_RANGE_M_FALLBACK = 50.0  # used only if the point cloud can't be read

CLASSIC_STAB_MIN_INLIERS = 15  # below this many affine-fit inliers, a frame-to-frame step is untrusted
OPTICAL_FLOW_FB_THRESHOLD_PX = 2.0
FFT_TARGET_BAND_HALFWIDTH_HZ = 0.15
REGISTRATION_ERROR_FLAG_THRESHOLD_PX = ROI_BOX_SIZE / 2
REGISTRATION_ERROR_FLAG_FRACTION_PCT = 15.0

REF_IDX = 352
# REF_IDX = 152
RANDOM_SEED = 42

# ====================================================================
# PART 1: DATA & MATH
# ====================================================================
def load_slam_data(json_path):
    with open(json_path, 'r') as f: data = json.load(f)
    K = np.array([[data['fl_x'], 0, data['cx']], [0, data['fl_y'], data['cy']], [0, 0, 1]], dtype=np.float32)
    return K, data['frames']

def extract_structural_plane(pcd_path, T_anch, K, pixel):
    print(f"\nLoading Global Point Cloud: {pcd_path}...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    pcd = pcd.voxel_down_sample(voxel_size=0.05) 
    
    R_anch = T_anch[:3, :3]
    C_w = T_anch[:3, 3]
    ray_c = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    ray_w = R_anch @ ray_c
    ray_w = ray_w / np.linalg.norm(ray_w)
    
    points = np.asarray(pcd.points)
    vecs = points - C_w
    t = np.dot(vecs, ray_w)
    dists = np.linalg.norm(vecs - np.outer(t, ray_w), axis=1)
    valid = (t > 0) & (dists < 0.1) 
    
    if np.any(valid):
        target_pt = points[valid][np.argmin(t[valid])] 
        dists_to_target = np.linalg.norm(points - target_pt, axis=1)
        pcd_filtered = pcd.select_by_index(np.where(dists_to_target < 1.0)[0])
    else:
        pcd_filtered = pcd
        
    plane_model, _ = pcd_filtered.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=1000)
    
    n_w = np.array(plane_model[:3])
    d_w = plane_model[3]
    
    if np.dot(n_w, C_w) + d_w < 0:
        n_w = -n_w
        d_w = -d_w
        
    n_c = R_anch.T @ n_w
    d_cam = -(np.dot(n_w, C_w) + d_w) 
    
    ray_c_gt = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    ray_w_gt = R_anch @ ray_c_gt
    s = d_cam / np.dot(n_w, ray_w_gt)
    P_gt_w = C_w + s * ray_w_gt
    
    return n_c, d_cam, P_gt_w, n_w, d_w

def compute_homography(K, T_anch, T_k, n_c, d_cam):
    R_anch = T_anch[:3, :3]
    t_anch = T_anch[:3, 3].reshape(3, 1)
    R_k = T_k[:3, :3]
    t_k = T_k[:3, 3].reshape(3, 1)
    
    R_rel = R_k.T @ R_anch
    t_rel = R_k.T @ (t_anch - t_k)
    
    H_k_metric = R_rel + (t_rel @ n_c.reshape(1, 3)) / d_cam
    H_k = K @ H_k_metric @ np.linalg.inv(K)
    
    if abs(H_k[2, 2]) > 1e-8:
        H_k = H_k / H_k[2, 2]
    return H_k

_scene_extent_cache = {}

def get_scene_bounds():
    if 'bounds' not in _scene_extent_cache:
        try:
            pts = np.asarray(o3d.io.read_point_cloud(PCD_FILE).points)
            bbox_min = pts.min(axis=0) - SCENE_EXTENT_MARGIN_M
            bbox_max = pts.max(axis=0) + SCENE_EXTENT_MARGIN_M
        except Exception:
            bbox_min = np.full(3, -MAX_PLAUSIBLE_RANGE_M_FALLBACK)
            bbox_max = np.full(3, MAX_PLAUSIBLE_RANGE_M_FALLBACK)
        _scene_extent_cache['bounds'] = (bbox_min, bbox_max)
        print(f"[SANITY BOUND] Raycast results outside the surveyed scene bounds "
              f"(LiDAR map bbox +/-{SCENE_EXTENT_MARGIN_M}m: {bbox_min} to {bbox_max}) "
              f"are rejected as invalid tracking.")
    return _scene_extent_cache['bounds']

def get_raycast_point_w(K, T_k, pixel, n_w, d_w):
    R_k = T_k[:3, :3]; t_k = T_k[:3, 3]
    ray_c = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    ray_w = R_k @ ray_c
    denom = np.dot(n_w, ray_w)
    if abs(denom) < 1e-6: return None
    P = t_k - ((np.dot(n_w, t_k) + d_w) / denom) * ray_w
    # denom being small-but-nonzero (grazing-incidence ray) still blows this up to a
    # physically meaningless point -- see get_scene_bounds() above.
    bbox_min, bbox_max = get_scene_bounds()
    if np.any(P < bbox_min) or np.any(P > bbox_max):
        return None
    return P

def get_pixel_in_frame(K, T_k, P_target_w):
    R_k = T_k[:3, :3]
    t_k = T_k[:3, 3]
    P_c = R_k.T @ (P_target_w - t_k)
    if P_c[2] <= 0: return None
    p_img = K @ P_c
    return (p_img[0] / p_img[2], p_img[1] / p_img[2])

def get_camera_depth(P_w, T_k):
    if P_w is None: return np.nan
    R_k = T_k[:3, :3]
    t_k = T_k[:3, 3]
    P_c = R_k.T @ (P_w - t_k)
    return abs(P_c[2])

def format_3d(pt): return f"[{pt[0]:.2f}, {pt[1]:.2f}, {pt[2]:.2f}]" if pt is not None else "[N/A]"

# ====================================================================
# PART 2: THE MASTER PROCESSING LOOP
# ====================================================================
def add_zoomed_inset(ax, img, x, y, roi_size, color='red', draw_grid=False):
    half_size = roi_size // 2
    h, w = img.shape[:2]
    x1, x2 = max(0, int(x) - half_size), min(w, int(x) + half_size)
    y1, y2 = max(0, int(y) - half_size), min(h, int(y) + half_size)
    
    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none', linestyle='dashed')
    ax.add_patch(rect)
    
    axins = ax.inset_axes([0.65, 0.65, 0.33, 0.33])
    axins.imshow(img)
    axins.set_xlim(x1, x2)
    axins.set_ylim(y2, y1) 
    axins.set_xticks([])
    axins.set_yticks([])
    
    if draw_grid:
        num_lines = 15 
        x_ticks = np.linspace(x1, x2, num_lines)
        y_ticks = np.linspace(y1, y2, num_lines)
        for xt in x_ticks: axins.axvline(xt, color='white', alpha=0.3, lw=0.8)
        for yt in y_ticks: axins.axhline(yt, color='white', alpha=0.3, lw=0.8)
        axins.text(x1 + (x2-x1)*0.03, y2 - (y2-y1)*0.05, "Dense Pixel Tensor", color='white', fontweight='bold', fontsize=10, bbox=dict(facecolor='black', alpha=0.6, pad=0.3))

    for spine in axins.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

def track_optical_flow(grays_list, ref_idx, start_pixel, fb_threshold=OPTICAL_FLOW_FB_THRESHOLD_PX):
    print("  -> Pre-computing Lucas-Kanade Optical Flow Baseline...")
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    of_coords = np.zeros((len(grays_list), 2))
    of_coords[ref_idx] = start_pixel
    n_rejected = 0

    def step(g_prev, g_curr, p0):
        nonlocal n_rejected
        p1, st, err = cv2.calcOpticalFlowPyrLK(g_prev, g_curr, p0, None, **lk_params)
        if st[0][0] != 1:
            n_rejected += 1
            return p0
        p0_check, st_back, _ = cv2.calcOpticalFlowPyrLK(g_curr, g_prev, p1, None, **lk_params)
        if st_back[0][0] != 1 or np.linalg.norm(p0_check[0][0] - p0[0][0]) > fb_threshold:
            n_rejected += 1
            return p0
        return p1

    p0 = np.array([[start_pixel]], dtype=np.float32)
    for i in range(ref_idx, len(grays_list) - 1):
        p0 = step(grays_list[i], grays_list[i+1], p0)
        of_coords[i+1] = p0[0][0]

    p0 = np.array([[start_pixel]], dtype=np.float32)
    for i in range(ref_idx, 0, -1):
        p0 = step(grays_list[i], grays_list[i-1], p0)
        of_coords[i-1] = p0[0][0]

    if n_rejected > 0:
        print(f"     Optical flow: rejected {n_rejected}/{len(grays_list) - 1} steps (lost track or"
              f" forward-backward mismatch > {fb_threshold}px) and held the last known-good position.")
    return of_coords

def track_ransac_homography(grays_list, ref_idx, min_inliers=RANSAC_MIN_INLIERS, reproj_thresh=RANSAC_REPROJ_THRESHOLD):
    print("  -> Pre-computing RANSAC Feature Homography Baseline (SIFT + RANSAC)...")
    detector = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    anchor_kp, anchor_des = detector.detectAndCompute(grays_list[ref_idx], None)

    homographies = [np.eye(3) for _ in grays_list]
    inlier_counts = [0] * len(grays_list)
    failed = [False] * len(grays_list)
    inlier_counts[ref_idx] = -1  # anchor frame itself: not applicable

    last_valid_H = np.eye(3)
    for i, gray in enumerate(grays_list):
        if i == ref_idx:
            continue

        kp, des = detector.detectAndCompute(gray, None)
        H_est, n_inliers = None, 0
        if des is not None and anchor_des is not None and len(des) >= 2:
            knn = matcher.knnMatch(anchor_des, des, k=2)
            good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
            if len(good) >= 4:
                src_pts = np.float32([anchor_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                H_candidate, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, reproj_thresh)
                if H_candidate is not None:
                    H_est = H_candidate
                    n_inliers = int(mask.sum())

        inlier_counts[i] = n_inliers
        if H_est is not None and n_inliers >= min_inliers:
            last_valid_H = H_est
            homographies[i] = H_est
        else:
            homographies[i] = last_valid_H.copy()
            failed[i] = True

    n_failed = sum(failed)
    print(f"     RANSAC homography: {n_failed}/{len(grays_list)} frames fell below {min_inliers} inliers (held last valid H).")
    return homographies, inlier_counts, failed

def _estimate_incremental_similarity(g_prev, g_curr, min_inliers=CLASSIC_STAB_MIN_INLIERS):
    feature_params = dict(maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=19)
    lk_params = dict(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    p0 = cv2.goodFeaturesToTrack(g_prev, mask=None, **feature_params)
    if p0 is None or len(p0) < 6:
        return np.eye(3), False
    p1, st, err = cv2.calcOpticalFlowPyrLK(g_prev, g_curr, p0, None, **lk_params)
    st = st.reshape(-1)
    good_prev, good_curr = p0[st == 1], p1[st == 1]
    if len(good_prev) < 6:
        return np.eye(3), False
    M, inliers = cv2.estimateAffinePartial2D(good_prev, good_curr, method=cv2.RANSAC)
    if M is None:
        return np.eye(3), False
    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if n_inliers < min_inliers:
        return np.eye(3), False
    return np.vstack([M, [0.0, 0.0, 1.0]]), True

def track_classic_stabilization(grays_list, ref_idx, smoothing_window=CLASSIC_STAB_SMOOTHING_WINDOW):
    print("  -> Pre-computing Classic Trajectory-Smoothing Stabilization Baseline...")
    n = len(grays_list)
    raw_homogs = [np.eye(3) for _ in range(n)]
    n_untrusted_steps = 0

    cum_fwd = np.eye(3)
    for i in range(ref_idx, n - 1):
        step, ok = _estimate_incremental_similarity(grays_list[i], grays_list[i + 1])
        if ok:
            cum_fwd = step @ cum_fwd
        else:
            n_untrusted_steps += 1  # hold cum_fwd: assume no motion rather than trust a low-confidence fit
        raw_homogs[i + 1] = cum_fwd.copy()

    cum_bwd = np.eye(3)
    for i in range(ref_idx, 0, -1):
        step, ok = _estimate_incremental_similarity(grays_list[i], grays_list[i - 1])
        if ok:
            cum_bwd = step @ cum_bwd
        else:
            n_untrusted_steps += 1
        raw_homogs[i - 1] = cum_bwd.copy()

    if n_untrusted_steps > 0:
        print(f"     Classic stabilization: {n_untrusted_steps}/{n - 1} frame-to-frame steps had"
              f" < {CLASSIC_STAB_MIN_INLIERS} inliers and held the previous position instead of"
              f" trusting a low-confidence fit.")

    dx = np.array([h[0, 2] for h in raw_homogs])
    dy = np.array([h[1, 2] for h in raw_homogs])
    da = np.array([np.arctan2(h[1, 0], h[0, 0]) for h in raw_homogs])

    def moving_average(x, w):
        w = min(w, len(x))
        if w < 3:
            return x.copy()
        pad = w // 2
        x_padded = np.pad(x, (pad, pad), mode='edge')
        return np.convolve(x_padded, np.ones(w) / w, mode='same')[pad:pad + len(x)]

    dx_s = moving_average(dx, smoothing_window)
    dy_s = moving_average(dy, smoothing_window)
    da_s = moving_average(da, smoothing_window)

    dx_s -= dx_s[ref_idx]
    dy_s -= dy_s[ref_idx]
    da_s -= da_s[ref_idx]

    smoothed_homogs = []
    for i in range(n):
        c, s = np.cos(da_s[i]), np.sin(da_s[i])
        smoothed_homogs.append(np.array([[c, -s, dx_s[i]], [s, c, dy_s[i]], [0.0, 0.0, 1.0]]))

    return smoothed_homogs

def build_master_pipeline(json_path, base_dir, downscale, ref_idx, n_c, d, n_w, d_w, P_gt):
    K, frames = load_slam_data(json_path)
    width, height = 1920, 1080 
    
    T_anch = np.array(frames[ref_idx]['transform_matrix'])
    gt_target_depth = get_camera_depth(P_gt, T_anch)
    
    new_w, new_h = int(width * downscale), int(height * downscale)
    stab_tensor, raw_tensor, ransac_tensor = [], [], []

    print(f"\nPhase 1: Generating full-resolution unconstrained frames...")
    raw_grays = []
    for i, frame_data in enumerate(frames):
        img_path = os.path.join(base_dir, frame_data['file_path'].replace('./', ''))
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw_grays.append(gray)

    print("Phase 2: Computing Optical Flow on Raw Video...")
    of_coords = track_optical_flow(raw_grays, ref_idx, CROSSHAIR_PIXEL)

    print("Phase 2.5: Computing RANSAC Feature Homography Baseline...")
    ransac_homogs, ransac_inlier_counts, ransac_failed = track_ransac_homography(raw_grays, ref_idx)

    print("Phase 3: Calculating 3D Mathematics and Packaging Plot Data...")
    of_signal = np.zeros(len(frames))
    raw_pixels, raw_3d_points, of_3d_points, ransac_3d_points, stab_3d_points = [], [], [], [], []
    raw_px_err, of_px_err, rt_px_err, vt_px_err = [], [], [], []

    plot_data = []
    frames_to_plot = [90, 180, 270, 360, 679]
    # frames_to_plot = [90, 100, 110, 303]
    anch_pixel_hom = np.array([CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], 1.0])

    for i, frame_data in enumerate(frames):
        T_k = np.array(frame_data['transform_matrix'])
        of_uv = of_coords[i]

        u, v = int(np.clip(of_uv[0], 0, width-1)), int(np.clip(of_uv[1], 0, height-1))
        of_signal[i] = raw_grays[i][v, u]

        raw_tensor.append(cv2.resize(raw_grays[i], (new_w, new_h), interpolation=cv2.INTER_AREA))
        H_k = compute_homography(K, T_anch, T_k, n_c, d)
        try:
            H_inv = np.linalg.inv(H_k)
        except np.linalg.LinAlgError:
            H_inv = np.linalg.pinv(H_k)

        stab_gray = cv2.warpPerspective(raw_grays[i], H_inv, (width, height), flags=cv2.INTER_LINEAR)
        stab_tensor.append(cv2.resize(stab_gray, (new_w, new_h), interpolation=cv2.INTER_AREA))

        H_ransac_k = ransac_homogs[i]
        try:
            H_ransac_inv = np.linalg.inv(H_ransac_k)
        except np.linalg.LinAlgError:
            H_ransac_inv = np.linalg.pinv(H_ransac_k)

        ransac_gray = cv2.warpPerspective(raw_grays[i], H_ransac_inv, (width, height), flags=cv2.INTER_LINEAR)
        ransac_tensor.append(cv2.resize(ransac_gray, (new_w, new_h), interpolation=cv2.INTER_AREA))

        tracked_uv = get_pixel_in_frame(K, T_k, P_gt)
        raw_pixels.append(tracked_uv)

        if tracked_uv is not None:
            raw_px_err.append(np.linalg.norm(np.array(CROSSHAIR_PIXEL) - np.array(tracked_uv)))
            of_px_err.append(np.linalg.norm(np.array(of_uv) - np.array(tracked_uv)))

            gt_hom = np.array([tracked_uv[0], tracked_uv[1], 1.0])
            gt_stab_hom = H_inv @ gt_hom
            if abs(gt_stab_hom[2]) > 1e-6:
                gt_stab_uv = (gt_stab_hom[0] / gt_stab_hom[2], gt_stab_hom[1] / gt_stab_hom[2])
                vt_px_err.append(np.linalg.norm(np.array(CROSSHAIR_PIXEL) - np.array(gt_stab_uv)))
            else:
                vt_px_err.append(np.nan)

            gt_ransac_stab_hom = H_ransac_inv @ gt_hom
            if abs(gt_ransac_stab_hom[2]) > 1e-6:
                gt_ransac_stab_uv = (gt_ransac_stab_hom[0] / gt_ransac_stab_hom[2], gt_ransac_stab_hom[1] / gt_ransac_stab_hom[2])
                rt_px_err.append(np.linalg.norm(np.array(CROSSHAIR_PIXEL) - np.array(gt_ransac_stab_uv)))
            else:
                rt_px_err.append(np.nan)
        else:
            raw_px_err.append(np.nan); of_px_err.append(np.nan); vt_px_err.append(np.nan); rt_px_err.append(np.nan)

        P_raw = get_raycast_point_w(K, T_k, CROSSHAIR_PIXEL, n_w, d_w)
        raw_3d_points.append(P_raw)
        raw_gate = abs(get_camera_depth(P_raw, T_k) - gt_target_depth) <= 0.6

        P_of = get_raycast_point_w(K, T_k, of_uv, n_w, d_w)
        of_3d_points.append(P_of)
        of_gate = abs(get_camera_depth(P_of, T_k) - gt_target_depth) <= 0.6

        u_raw_hom = H_k @ anch_pixel_hom
        if abs(u_raw_hom[2]) > 1e-4:
            P_stab = get_raycast_point_w(K, T_k, u_raw_hom[:2] / u_raw_hom[2], n_w, d_w)
        else:
            P_stab = None
        stab_3d_points.append(P_stab)
        stab_gate = abs(get_camera_depth(P_stab, T_k) - gt_target_depth) <= 0.6
        u_ransac_hom = H_ransac_k @ anch_pixel_hom
        if abs(u_ransac_hom[2]) > 1e-4:
            P_ransac = get_raycast_point_w(K, T_k, u_ransac_hom[:2] / u_ransac_hom[2], n_w, d_w)
        else:
            P_ransac = None
        ransac_3d_points.append(P_ransac)
        ransac_gate = abs(get_camera_depth(P_ransac, T_k) - gt_target_depth) <= 0.6

        if i in frames_to_plot:
            img_path = os.path.join(base_dir, frame_data['file_path'].replace('./', ''))
            img = cv2.imread(img_path)
            raw_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            of_rgb = raw_rgb.copy()

            tracked_uv_int = (int(tracked_uv[0]), int(tracked_uv[1])) if tracked_uv is not None else None

            if tracked_uv_int is not None:
                cv2.drawMarker(raw_rgb, tracked_uv_int, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=50)
            cv2.drawMarker(raw_rgb, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)

            if tracked_uv_int is not None:
                cv2.drawMarker(of_rgb, tracked_uv_int, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=50)
            cv2.drawMarker(of_rgb, (int(of_uv[0]), int(of_uv[1])), (255, 165, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)

            stab_rgb = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), H_inv, (width, height), flags=cv2.INTER_LINEAR)
            ransac_rgb = cv2.warpPerspective(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), H_ransac_inv, (width, height), flags=cv2.INTER_LINEAR)

            if tracked_uv is not None:
                gt_hom = np.array([tracked_uv[0], tracked_uv[1], 1.0])
                gt_stab_hom = H_inv @ gt_hom
                if abs(gt_stab_hom[2]) > 1e-6:
                    gt_stab_uv_int = (int(gt_stab_hom[0] / gt_stab_hom[2]), int(gt_stab_hom[1] / gt_stab_hom[2]))
                    cv2.drawMarker(stab_rgb, gt_stab_uv_int, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=50)

                gt_ransac_stab_hom = H_ransac_inv @ gt_hom
                if abs(gt_ransac_stab_hom[2]) > 1e-6:
                    gt_ransac_stab_uv_int = (int(gt_ransac_stab_hom[0] / gt_ransac_stab_hom[2]), int(gt_ransac_stab_hom[1] / gt_ransac_stab_hom[2]))
                    cv2.drawMarker(ransac_rgb, gt_ransac_stab_uv_int, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=50)

            of_hom = np.array([of_uv[0], of_uv[1], 1.0])
            of_stab_hom = H_inv @ of_hom
            if abs(of_stab_hom[2]) > 1e-6:
                of_stab_uv_int = (int(of_stab_hom[0] / of_stab_hom[2]), int(of_stab_hom[1] / of_stab_hom[2]))
                cv2.drawMarker(stab_rgb, of_stab_uv_int, (255, 165, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)

            cv2.drawMarker(stab_rgb, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)
            cv2.drawMarker(ransac_rgb, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)

            plot_data.append((i, raw_rgb, of_rgb, ransac_rgb, stab_rgb, tracked_uv_int, P_raw, of_uv, P_of, P_ransac, P_stab, raw_gate, of_gate, ransac_gate, stab_gate, ransac_inlier_counts[i], ransac_failed[i]))

    return (np.stack(raw_tensor, axis=0), np.stack(stab_tensor, axis=0), np.stack(ransac_tensor, axis=0),
            of_signal, frames, plot_data, raw_pixels,
            raw_3d_points, of_3d_points, ransac_3d_points, stab_3d_points,
            raw_px_err, of_px_err, rt_px_err, vt_px_err, of_coords,
            ransac_inlier_counts, ransac_failed, ransac_homogs)

# ====================================================================
# PART 3.5: REAL-DATA NYQUIST MODAL ANALYSIS FIGURE (Fig 8)
# ====================================================================
def generate_nyquist_modal_figure(time_axis, real_signal, freqs, real_fft, fps, target_freq, save_dir):
    print("\nGenerating Figure 8: Nyquist-Shannon Validation (Real Empirical Data)...")
    nyquist_limit = fps / 2.0
    min_fps_required = target_freq * 2.0
    oversample_ratio = fps / min_fps_required

    max_idx = int(5.0 * fps)
    if max_idx > len(time_axis): max_idx = len(time_axis)
    
    t_slice = time_axis[:max_idx]
    sig_slice = real_signal[:max_idx]
    
    sig_slice = (sig_slice - np.mean(sig_slice)) / (np.std(sig_slice) + 1e-6)

    t_continuous = np.linspace(t_slice[0], t_slice[-1], 500)
    sig_continuous = np.interp(t_continuous, t_slice, sig_slice)
    
    plt.rcParams.update({'mathtext.fontset': 'cm', 'font.size': 12, 'font.family': 'sans-serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.canvas.manager.set_window_title('Figure 8: Nyquist Modal Analysis')
    fig.suptitle('Nyquist-Shannon Validation for Sub-Pixel Modal Analysis (Empirical Data)', fontsize=18, fontweight='bold')

    ax1.plot(t_continuous, sig_continuous, color='gray', linestyle='-', linewidth=2, alpha=0.5, label='Interpolated Physical Vibration')
    ax1.stem(t_slice, sig_slice, linefmt='b-', markerfmt='bo', basefmt='k-', label=f'Camera Sampling ({fps} FPS)')
    
    ax1.set_title('(a) Temporal Oversampling Margin (First 5 Seconds)', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Time (seconds)', fontweight='bold')
    ax1.set_ylabel('Normalized Pixel Displacement', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='upper right')
    
    text_a = (f"Nyquist Requirement: {min_fps_required:.2f} FPS\n"
              f"Actual Hardware: {fps} FPS\n"
              f"Oversampling Ratio: {oversample_ratio:.1f}x")
    ax1.text(0.03, 0.04, text_a, transform=ax1.transAxes, color='black', fontweight='bold', 
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    ax2.plot(freqs, real_fft, color='forestgreen', linewidth=2.5, label='Extracted 3D Gate Spectrum')
    ax2.axvline(target_freq, color='red', linestyle='-', linewidth=2, label=f'Target Mode ($f_{{target}}$ = {target_freq} Hz)')
    ax2.axvline(nyquist_limit, color='black', linestyle='--', linewidth=2, label=f'Nyquist Limit ($f_s/2$ = {nyquist_limit} Hz)')
    ax2.axvspan(0, nyquist_limit, color='palegreen', alpha=0.15, label='Valid Sub-Nyquist Spectral Zone')

    ax2.set_title('(b) Empirical Spectral Extraction Bounds', fontweight='bold', fontsize=14)
    ax2.set_xlabel('Frequency (Hz)', fontweight='bold')
    ax2.set_ylabel('FFT Amplitude', fontweight='bold')
    ax2.set_xlim(0, 11)
    ax2.set_ylim(0, max(real_fft) * 1.2)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='upper right')

    formula_text = r"Nyquist Condition: $f_s \geq 2f_{target}$"
    ax2.text(0.5, 0.85, formula_text, transform=ax2.transAxes, ha='center', fontsize=14, color='black',
             bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4'))

    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    fig.savefig(os.path.join(save_dir, 'Figure_8_Nyquist_Modal_Analysis.pdf'), dpi=300, bbox_inches='tight')

# ====================================================================
# PART 3: MULTI-FIGURE PLOTTING (Figs 1, 2, 3)
# ====================================================================
def render_plots(raw_tensor, stab_tensor, ransac_tensor, of_signal, plot_data, frames, ref_idx, P_gt,
                  raw_pixels, raw_3d, of_3d, ransac_3d, stab_3d,
                  raw_px_err, of_px_err, rt_px_err, vt_px_err, fps, downscale):
    plt.rcParams.update({'mathtext.fontset': 'cm'})

    fig1 = plt.figure(figsize=(32, 20))
    fig1.canvas.manager.set_window_title('Figure 1: Visual Proof Grid')
    gs = fig1.add_gridspec(3, 6, width_ratios=[1.3, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.4], wspace=0.05, hspace=0.25)

    ax_traj = fig1.add_subplot(gs[0:2, 0])
    traj_x = [np.array(f['transform_matrix'])[0, 3] for f in frames]
    traj_y = [np.array(f['transform_matrix'])[1, 3] for f in frames]
    ax_traj.plot(traj_x, traj_y, color='gray', linewidth=2)
    ax_traj.scatter(traj_x[ref_idx], traj_y[ref_idx], color='red', s=200, marker='*')

    for (idx, *_) in plot_data:
        ax_traj.scatter(traj_x[idx], traj_y[idx], color='blue', s=80)

    ax_traj.set_title("Backend SLAM Trajectory", fontsize=16, fontweight='bold')
    ax_traj.set_aspect('equal')

    ax_anchor = fig1.add_subplot(gs[2, 0])
    anch_path = os.path.join(DATA_DIR, frames[ref_idx]['file_path'].replace('./', ''))
    anch_img = cv2.cvtColor(cv2.imread(anch_path), cv2.COLOR_BGR2RGB)
    cv2.drawMarker(anch_img, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)
    ax_anchor.imshow(anch_img)
    add_zoomed_inset(ax_anchor, anch_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE)
    ax_anchor.set_title(f"Stabilization Anchor ($t_{{anch}} = {ref_idx}$)\nTarget Pixel: {CROSSHAIR_PIXEL}", fontsize=15, fontweight='bold', color='darkred')
    ax_anchor.text(0.5, -0.15, f"Ground Truth 3D Target:\n{format_3d(P_gt)}", color='darkred', fontweight='bold', fontsize=14, ha='center', transform=ax_anchor.transAxes)
    ax_anchor.text(0.5, -0.40, "(Red Cross: Eulerian Fixed | Orange Cross: OF Tracker | Green Cross: Ground Truth)", color='black', fontweight='bold', fontsize=12, ha='center', transform=ax_anchor.transAxes)
    ax_anchor.axis('off')
    for col, (idx, raw_img, of_img, ransac_img, stab_img, tracked_uv_int, P_raw, of_uv, P_of, P_ransac, P_stab, raw_gate, of_gate, ransac_gate, stab_gate, ransac_inliers, ransac_held) in enumerate(plot_data, start=1):
        prefix = r". . . $\rightarrow$ " if col == len(plot_data) else ""
        display_idx = "last" if col == len(plot_data) else str(idx)

        r_err = np.linalg.norm(P_raw - P_gt) if P_raw is not None else np.nan
        o_err = np.linalg.norm(P_of - P_gt) if P_of is not None else np.nan
        s_err = np.linalg.norm(P_stab - P_gt) if P_stab is not None else np.nan

        r_err_str = f"(\u0394 {r_err:.2f}m Error)" if not np.isnan(r_err) else "(N/A)"
        o_err_str = f"(\u0394 {o_err:.2f}m Error)" if not np.isnan(o_err) else "(N/A)"
        s_err_str = f"(\u0394 {s_err:.2f}m Error)" if not np.isnan(s_err) else "(N/A)"

        raw_gate_txt = "LiDAR Gate: ACCEPTED" if raw_gate else "LiDAR Gate: REJECTED"
        of_gate_txt = "LiDAR Gate: ACCEPTED" if of_gate else "LiDAR Gate: REJECTED"
        stab_gate_txt = "LiDAR Gate: ACCEPTED" if stab_gate else "LiDAR Gate: REJECTED"

        ax_raw = fig1.add_subplot(gs[0, col])
        ax_raw.imshow(raw_img)
        add_zoomed_inset(ax_raw, raw_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE, 'red')
        ax_raw.text(0.5, 1.25, r_err_str, color='red', fontweight='bold', fontsize=13, ha='center', transform=ax_raw.transAxes)
        ax_raw.text(0.5, 1.15, f"3D Hit: {format_3d(P_raw)}", fontsize=11, ha='center', transform=ax_raw.transAxes)
        ax_raw.text(0.5, 1.05, f"{prefix}Unconstrained Video ($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_raw.transAxes)
        ax_raw.text(0.5, -0.15, raw_gate_txt, color='green' if raw_gate else 'red', fontweight='bold', fontsize=12, ha='center', transform=ax_raw.transAxes)
        ax_raw.axis('off')

        ax_of = fig1.add_subplot(gs[1, col])
        ax_of.imshow(of_img)
        add_zoomed_inset(ax_of, of_img, of_uv[0], of_uv[1], ROI_BOX_SIZE, 'orange')
        ax_of.text(0.5, 1.25, o_err_str, color='orange', fontweight='bold', fontsize=13, ha='center', transform=ax_of.transAxes)
        ax_of.text(0.5, 1.15, f"3D Hit: {format_3d(P_of)}", fontsize=11, ha='center', transform=ax_of.transAxes)
        ax_of.text(0.5, 1.05, f"{prefix}Optical Flow Baseline ($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_of.transAxes)
        ax_of.text(0.5, -0.15, of_gate_txt, color='green' if of_gate else 'red', fontweight='bold', fontsize=12, ha='center', transform=ax_of.transAxes)
        ax_of.axis('off')

        ax_stab = fig1.add_subplot(gs[2, col])
        ax_stab.imshow(stab_img)
        add_zoomed_inset(ax_stab, stab_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE, 'green')
        ax_stab.text(0.5, 1.25, s_err_str, color='green', fontweight='bold', fontsize=13, ha='center', transform=ax_stab.transAxes)
        ax_stab.text(0.5, 1.15, f"3D Hit: {format_3d(P_stab)}", fontsize=11, ha='center', transform=ax_stab.transAxes)
        ax_stab.text(0.5, 1.05, f"{prefix}Virtual Tripod Stabilization ($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_stab.transAxes)
        ax_stab.text(0.5, -0.15, stab_gate_txt, color='green' if stab_gate else 'red', fontweight='bold', fontsize=12, ha='center', transform=ax_stab.transAxes)
        ax_stab.axis('off')

    fig1.tight_layout(pad=2.0)

    time_axis = np.arange(len(raw_3d)) / fps
    raw_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in raw_3d]
    of_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in of_3d]
    ransac_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in ransac_3d]
    stab_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in stab_3d]

    # FFT data (Figure 3 is produced separately by render_merged_fft_spectrum(),
    # combining this data with Figure Set B's -- same pattern as Figure 2's merge)
    T = raw_tensor.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0/fps)
    window = np.hanning(T)
    sx, sy = int(CROSSHAIR_PIXEL[0] * downscale), int(CROSSHAIR_PIXEL[1] * downscale)

    raw_sig = raw_tensor[:, sy, sx]
    stab_sig = stab_tensor[:, sy, sx]
    ransac_sig = ransac_tensor[:, sy, sx]

    raw_fft = np.abs(np.fft.rfft((raw_sig - np.mean(raw_sig)) * window))
    of_fft = np.abs(np.fft.rfft((of_signal - np.mean(of_signal)) * window))
    ransac_fft = np.abs(np.fft.rfft((ransac_sig - np.mean(ransac_sig)) * window))
    stab_fft = np.abs(np.fft.rfft((stab_sig - np.mean(stab_sig)) * window))

    # --- CALLING THE NEW REAL-DATA NYQUIST PLOT HERE ---
    generate_nyquist_modal_figure(time_axis, stab_sig, freqs, stab_fft, fps, TARGET_FREQ_HZ, DATA_DIR)

    print("\nExporting raw numerical data to CSV files...")
    df_tracking = pd.DataFrame({
        'Time_Seconds': time_axis,
        'Unconstrained_3D_Error_m': raw_drift, 'OpticalFlow_3D_Error_m': of_drift,
        'RANSACHomography_3D_Error_m': ransac_drift, 'VirtualTripod_3D_Error_m': stab_drift,
        'Unconstrained_2D_Error_px': raw_px_err, 'OpticalFlow_2D_Error_px': of_px_err,
        'RANSACHomography_2D_Error_px': rt_px_err, 'VirtualTripod_2D_Error_px': vt_px_err,
    })
    df_tracking.to_csv(os.path.join(DATA_DIR, 'Figure2_Tracking_Errors_Data.csv'), index=False)
    df_fft = pd.DataFrame({
        'Frequency_Hz': freqs,
        'Unconstrained_FFT_Amplitude': raw_fft, 'OpticalFlow_FFT_Amplitude': of_fft,
        'RANSACHomography_FFT_Amplitude': ransac_fft, 'VirtualTripod_FFT_Amplitude': stab_fft,
    })
    df_fft.to_csv(os.path.join(DATA_DIR, 'Figure3_FFT_Spectrum_Data.csv'), index=False)

    fig1.savefig(os.path.join(DATA_DIR, 'Figure_1_Tracking_Grid.pdf'), dpi=300, bbox_inches='tight')

    return {
        'raw_drift': raw_drift, 'of_drift': of_drift, 'ransac_drift': ransac_drift, 'stab_drift': stab_drift,
        'raw_px_err': raw_px_err, 'of_px_err': of_px_err, 'rt_px_err': rt_px_err, 'vt_px_err': vt_px_err,
        'freqs': freqs, 'raw_fft': raw_fft, 'of_fft': of_fft, 'ransac_fft': ransac_fft, 'stab_fft': stab_fft,
    }

# ====================================================================
# PART 3.6: ABLATION FIGURE SET B -- STABILIZATION METHOD COMPARISON
# ====================================================================
def build_ablation_pipeline_b(json_path, base_dir, downscale, ref_idx, n_c, d, n_w, d_w, P_gt):
    K, frames = load_slam_data(json_path)
    width, height = 1920, 1080

    T_anch = np.array(frames[ref_idx]['transform_matrix'])
    gt_target_depth = get_camera_depth(P_gt, T_anch)

    new_w, new_h = int(width * downscale), int(height * downscale)
    stab_tensor, naive_tensor, classic_tensor = [], [], []

    print(f"\n[Figure Set B] Phase 1: Loading full-resolution frames...")
    raw_grays = []
    for frame_data in frames:
        img_path = os.path.join(base_dir, frame_data['file_path'].replace('./', ''))
        img = cv2.imread(img_path)
        raw_grays.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    classic_homogs = track_classic_stabilization(raw_grays, ref_idx)

    # Naive plane: fronto-parallel to the ANCHOR camera (n_c = camera Z-axis) at a fixed
    # GUESSED depth -- zero LiDAR geometry input, unlike n_c/d passed in for "ours".
    n_c_naive = np.array([0.0, 0.0, 1.0])
    d_naive = NAIVE_PLANE_ASSUMED_DEPTH

    print("[Figure Set B] Phase 2: Calculating 3D Mathematics and Packaging Plot Data...")
    stab_3d, naive_3d, classic_3d = [], [], []
    vt_px_err, naive_px_err, classic_px_err = [], [], []
    plot_data_b = []
    frames_to_plot = [90, 180, 270, 360, 679]
    # frames_to_plot = [90, 100, 110, 303]
    anch_pixel_hom = np.array([CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], 1.0])

    for i, frame_data in enumerate(frames):
        T_k = np.array(frame_data['transform_matrix'])

        H_stab_k = compute_homography(K, T_anch, T_k, n_c, d)
        H_naive_k = compute_homography(K, T_anch, T_k, n_c_naive, d_naive)
        H_classic_k = classic_homogs[i]

        def _inv(H):
            try:
                return np.linalg.inv(H)
            except np.linalg.LinAlgError:
                return np.linalg.pinv(H)

        H_stab_inv, H_naive_inv, H_classic_inv = _inv(H_stab_k), _inv(H_naive_k), _inv(H_classic_k)

        stab_tensor.append(cv2.resize(cv2.warpPerspective(raw_grays[i], H_stab_inv, (width, height), flags=cv2.INTER_LINEAR), (new_w, new_h), interpolation=cv2.INTER_AREA))
        naive_tensor.append(cv2.resize(cv2.warpPerspective(raw_grays[i], H_naive_inv, (width, height), flags=cv2.INTER_LINEAR), (new_w, new_h), interpolation=cv2.INTER_AREA))
        classic_tensor.append(cv2.resize(cv2.warpPerspective(raw_grays[i], H_classic_inv, (width, height), flags=cv2.INTER_LINEAR), (new_w, new_h), interpolation=cv2.INTER_AREA))

        tracked_uv = get_pixel_in_frame(K, T_k, P_gt)

        def _post_warp_err(H_inv):
            if tracked_uv is None:
                return np.nan
            gt_hom = np.array([tracked_uv[0], tracked_uv[1], 1.0])
            gt_stab_hom = H_inv @ gt_hom
            if abs(gt_stab_hom[2]) <= 1e-6:
                return np.nan
            gt_stab_uv = (gt_stab_hom[0] / gt_stab_hom[2], gt_stab_hom[1] / gt_stab_hom[2])
            return np.linalg.norm(np.array(CROSSHAIR_PIXEL) - np.array(gt_stab_uv))

        vt_px_err.append(_post_warp_err(H_stab_inv))
        naive_px_err.append(_post_warp_err(H_naive_inv))
        classic_px_err.append(_post_warp_err(H_classic_inv))

        def _raycast_via_our_plane(H_fwd):
            u_hom = H_fwd @ anch_pixel_hom
            if abs(u_hom[2]) <= 1e-4:
                return None
            return get_raycast_point_w(K, T_k, u_hom[:2] / u_hom[2], n_w, d_w)

        P_stab = _raycast_via_our_plane(H_stab_k)
        P_naive = _raycast_via_our_plane(H_naive_k)
        P_classic = _raycast_via_our_plane(H_classic_k)
        stab_3d.append(P_stab); naive_3d.append(P_naive); classic_3d.append(P_classic)

        stab_gate = abs(get_camera_depth(P_stab, T_k) - gt_target_depth) <= 0.6
        naive_gate = abs(get_camera_depth(P_naive, T_k) - gt_target_depth) <= 0.6
        classic_gate = abs(get_camera_depth(P_classic, T_k) - gt_target_depth) <= 0.6

        if i in frames_to_plot:
            img = cv2.imread(os.path.join(base_dir, frame_data['file_path'].replace('./', '')))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tracked_uv_int = (int(tracked_uv[0]), int(tracked_uv[1])) if tracked_uv is not None else None

            def _stab_rgb(H_inv):
                out = cv2.warpPerspective(img_rgb, H_inv, (width, height), flags=cv2.INTER_LINEAR)
                if tracked_uv is not None:
                    gt_hom = np.array([tracked_uv[0], tracked_uv[1], 1.0])
                    gt_stab_hom = H_inv @ gt_hom
                    if abs(gt_stab_hom[2]) > 1e-6:
                        pt = (int(gt_stab_hom[0] / gt_stab_hom[2]), int(gt_stab_hom[1] / gt_stab_hom[2]))
                        cv2.drawMarker(out, pt, (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=5, markerSize=50)
                cv2.drawMarker(out, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)
                return out

            naive_rgb = _stab_rgb(H_naive_inv)
            classic_rgb = _stab_rgb(H_classic_inv)
            stab_rgb = _stab_rgb(H_stab_inv)

            plot_data_b.append((i, naive_rgb, classic_rgb, stab_rgb, P_naive, P_classic, P_stab, naive_gate, classic_gate, stab_gate))

    return (np.stack(naive_tensor, axis=0), np.stack(classic_tensor, axis=0), np.stack(stab_tensor, axis=0),
            frames, plot_data_b, naive_3d, classic_3d, stab_3d,
            naive_px_err, classic_px_err, vt_px_err, classic_homogs)

def render_plots_b(plot_data, naive_tensor, classic_tensor, stab_tensor, plot_data_b, frames, ref_idx, P_gt,
                    naive_3d, classic_3d, stab_3d, naive_px_err, classic_px_err, vt_px_err, fps, downscale):
    plt.rcParams.update({'mathtext.fontset': 'cm'})

    fig1b = plt.figure(figsize=(32, 20))
    fig1b.canvas.manager.set_window_title('Figure 1B: Additional Stabilization Baselines')
    gs = fig1b.add_gridspec(3, 6, width_ratios=[1.3, 1, 1, 1, 1, 1], height_ratios=[1, 1, 1.4], wspace=0.05, hspace=0.25)

    ax_traj = fig1b.add_subplot(gs[0:2, 0])
    traj_x = [np.array(f['transform_matrix'])[0, 3] for f in frames]
    traj_y = [np.array(f['transform_matrix'])[1, 3] for f in frames]
    ax_traj.plot(traj_x, traj_y, color='gray', linewidth=2)
    ax_traj.scatter(traj_x[ref_idx], traj_y[ref_idx], color='red', s=200, marker='*')
    for (idx, *_) in plot_data_b:
        ax_traj.scatter(traj_x[idx], traj_y[idx], color='blue', s=80)
    ax_traj.set_title("Backend SLAM Trajectory", fontsize=16, fontweight='bold')
    ax_traj.set_aspect('equal')

    ax_anchor = fig1b.add_subplot(gs[2, 0])
    anch_path = os.path.join(DATA_DIR, frames[ref_idx]['file_path'].replace('./', ''))
    anch_img = cv2.cvtColor(cv2.imread(anch_path), cv2.COLOR_BGR2RGB)
    cv2.drawMarker(anch_img, CROSSHAIR_PIXEL, (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=3, markerSize=40)
    ax_anchor.imshow(anch_img)
    add_zoomed_inset(ax_anchor, anch_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE)
    ax_anchor.set_title(f"Stabilization Anchor ($t_{{anch}} = {ref_idx}$)\nTarget Pixel: {CROSSHAIR_PIXEL}", fontsize=15, fontweight='bold', color='darkred')
    ax_anchor.text(0.5, -0.15, f"Ground Truth 3D Target:\n{format_3d(P_gt)}", color='darkred', fontweight='bold', fontsize=14, ha='center', transform=ax_anchor.transAxes)
    ax_anchor.text(0.5, -0.40, "(Purple: RANSAC Homography | Goldenrod: Naive SLAM-only Plane | Teal: Classic Stabilizer)", color='black', fontweight='bold', fontsize=12, ha='center', transform=ax_anchor.transAxes)
    ax_anchor.axis('off')

    for col, (ransac_entry, naive_entry) in enumerate(zip(plot_data, plot_data_b), start=1):
        idx, _raw_img, _of_img, ransac_img, _stab_img_a, _tracked_uv_int, _P_raw, _of_uv, _P_of, P_ransac, _P_stab_a, _raw_gate, _of_gate, ransac_gate, _stab_gate_a, ransac_inliers, ransac_held = ransac_entry
        _idx_b, naive_img, classic_img, stab_img, P_naive, P_classic, P_stab, naive_gate, classic_gate, stab_gate = naive_entry

        prefix = r". . . $\rightarrow$ " if col == len(plot_data_b) else ""
        display_idx = "last" if col == len(plot_data_b) else str(idx)

        ra_err = np.linalg.norm(P_ransac - P_gt) if P_ransac is not None else np.nan
        n_err = np.linalg.norm(P_naive - P_gt) if P_naive is not None else np.nan
        c_err = np.linalg.norm(P_classic - P_gt) if P_classic is not None else np.nan

        ra_err_str = f"(Δ {ra_err:.2f}m Error)" if not np.isnan(ra_err) else "(N/A)"
        n_err_str = f"(Δ {n_err:.2f}m Error)" if not np.isnan(n_err) else "(N/A)"
        c_err_str = f"(Δ {c_err:.2f}m Error)" if not np.isnan(c_err) else "(N/A)"
        ransac_diag_txt = f"Inliers: {ransac_inliers}" + (" [HELD - LOW TEXTURE]" if ransac_held else "")

        ax_ransac = fig1b.add_subplot(gs[0, col])
        ax_ransac.imshow(ransac_img)
        add_zoomed_inset(ax_ransac, ransac_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE, 'purple')
        ax_ransac.text(0.5, 1.25, ra_err_str, color='purple', fontweight='bold', fontsize=13, ha='center', transform=ax_ransac.transAxes)
        ax_ransac.text(0.5, 1.15, f"3D Hit: {format_3d(P_ransac)}", fontsize=11, ha='center', transform=ax_ransac.transAxes)
        ax_ransac.text(0.5, 1.05, f"{prefix}RANSAC Homography Baseline ($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_ransac.transAxes)
        ax_ransac.text(0.5, -0.15, ransac_diag_txt, color='darkred' if ransac_held else 'purple', fontweight='bold', fontsize=12, ha='center', transform=ax_ransac.transAxes)
        ax_ransac.axis('off')

        ax_naive = fig1b.add_subplot(gs[2, col])
        ax_naive.imshow(naive_img)
        add_zoomed_inset(ax_naive, naive_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE, 'goldenrod')
        ax_naive.text(0.5, 1.25, n_err_str, color='goldenrod', fontweight='bold', fontsize=13, ha='center', transform=ax_naive.transAxes)
        ax_naive.text(0.5, 1.15, f"3D Hit: {format_3d(P_naive)}", fontsize=11, ha='center', transform=ax_naive.transAxes)
        ax_naive.text(0.5, 1.05, f"{prefix}SLAM w/o LiDAR gating Baseline($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_naive.transAxes)
        ax_naive.text(0.5, -0.15, "LiDAR Gate: ACCEPTED" if naive_gate else "LiDAR Gate: REJECTED", color='green' if naive_gate else 'red', fontweight='bold', fontsize=12, ha='center', transform=ax_naive.transAxes)
        ax_naive.axis('off')

        ax_classic = fig1b.add_subplot(gs[1, col])
        ax_classic.imshow(classic_img)
        add_zoomed_inset(ax_classic, classic_img, CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], ROI_BOX_SIZE, 'teal')
        ax_classic.text(0.5, 1.25, c_err_str, color='teal', fontweight='bold', fontsize=13, ha='center', transform=ax_classic.transAxes)
        ax_classic.text(0.5, 1.15, f"3D Hit: {format_3d(P_classic)}", fontsize=11, ha='center', transform=ax_classic.transAxes)
        ax_classic.text(0.5, 1.05, f"{prefix}LK+RANSAC Baseline ($t_{{{display_idx}}}$)", fontsize=13, ha='center', transform=ax_classic.transAxes)
        ax_classic.text(0.5, -0.15, "LiDAR Gate: ACCEPTED" if classic_gate else "LiDAR Gate: REJECTED", color='green' if classic_gate else 'red', fontweight='bold', fontsize=12, ha='center', transform=ax_classic.transAxes)
        ax_classic.axis('off')

    fig1b.tight_layout(pad=2.0)

    # FIG 3B: Plot Generation 
    time_axis = np.arange(len(naive_3d)) / fps
    naive_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in naive_3d]
    classic_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in classic_3d]
    stab_drift = [np.linalg.norm(p - P_gt) if p is not None else np.nan for p in stab_3d]

    T = naive_tensor.shape[0]
    freqs = np.fft.rfftfreq(T, d=1.0 / fps)
    window = np.hanning(T)
    sx, sy = int(CROSSHAIR_PIXEL[0] * downscale), int(CROSSHAIR_PIXEL[1] * downscale)

    naive_sig = naive_tensor[:, sy, sx]
    classic_sig = classic_tensor[:, sy, sx]
    stab_sig = stab_tensor[:, sy, sx]

    naive_fft = np.abs(np.fft.rfft((naive_sig - np.mean(naive_sig)) * window))
    classic_fft = np.abs(np.fft.rfft((classic_sig - np.mean(classic_sig)) * window))
    stab_fft = np.abs(np.fft.rfft((stab_sig - np.mean(stab_sig)) * window))

    print("\n[Figure Set B] Exporting raw numerical data to CSV files...")
    df_tracking_b = pd.DataFrame({
        'Time_Seconds': time_axis,
        'NaiveSLAMOnly_3D_Error_m': naive_drift, 'ClassicStabilization_3D_Error_m': classic_drift, 'VirtualTripod_3D_Error_m': stab_drift,
        'NaiveSLAMOnly_2D_Error_px': naive_px_err, 'ClassicStabilization_2D_Error_px': classic_px_err, 'VirtualTripod_2D_Error_px': vt_px_err,
    })
    df_tracking_b.to_csv(os.path.join(DATA_DIR, 'Figure2B_Tracking_Errors_Data.csv'), index=False)
    df_fft_b = pd.DataFrame({
        'Frequency_Hz': freqs,
        'NaiveSLAMOnly_FFT_Amplitude': naive_fft, 'ClassicStabilization_FFT_Amplitude': classic_fft, 'VirtualTripod_FFT_Amplitude': stab_fft,
    })
    df_fft_b.to_csv(os.path.join(DATA_DIR, 'Figure3B_FFT_Spectrum_Data.csv'), index=False)

    fig1b.savefig(os.path.join(DATA_DIR, 'Figure_1B_Additional_Baselines_Grid.pdf'), dpi=300, bbox_inches='tight')

    return {
        'naive_drift': naive_drift, 'classic_drift': classic_drift, 'stab_drift': stab_drift,
        'naive_px_err': naive_px_err, 'classic_px_err': classic_px_err, 'vt_px_err': vt_px_err,
        'freqs': freqs, 'naive_fft': naive_fft, 'classic_fft': classic_fft, 'stab_fft': stab_fft,
    }

def _plot_series_with_scale_guard(ax, time_axis, series, ylabel, title, unit, outlier_factor=OUTLIER_SCALE_FACTOR):
    peaks = np.array([np.nanmax(np.asarray(y, dtype=float)) if np.any(np.isfinite(y)) else 0.0 for y, *_ in series])
    median_peak = np.median(peaks) if len(peaks) else 0.0
    is_outlier = peaks > max(outlier_factor * median_peak, median_peak + 1e-9)

    if not np.any(~is_outlier):
        is_outlier[:] = False  # degenerate: everything "extreme" relative to itself -> just autoscale

    y_max = (peaks[~is_outlier].max() * 1.3) if np.any(~is_outlier) else max(peaks.max(), 1e-6) * 1.15
    y_max = max(y_max, 1e-6)

    outlier_report = []
    for (y, color, label, lw, alpha), peak, outlier in zip(series, peaks, is_outlier):
        ax.plot(time_axis, y, color=color, label=label, linewidth=lw, alpha=alpha)
        if outlier:
            short_label = label.split(' (')[0]
            outlier_report.append((short_label, float(peak)))

    ax.set_ylim(0, y_max)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper left', fontsize=10)

    if outlier_report:
        note = "Off-chart peak(s): " + " | ".join(f"{lbl} {pk:.1f}{unit}" for lbl, pk in outlier_report)
        ax.text(0.99, 0.02, note, transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
                color='dimgray', bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.3'))

    return outlier_report

def render_merged_tracking_stability(drift_metrics, drift_metrics_b, fps, save_dir=DATA_DIR):
    n = len(drift_metrics['raw_drift'])
    time_axis = np.arange(n) / fps

    fig2, (ax_2d, ax_3d) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig2.canvas.manager.set_window_title('Figure 2: Spatiotemporal Tracking Stability (All Methods)')

    series_2d = [
        (drift_metrics['raw_px_err'], 'red', 'Unconstrained Pixel', 2, 0.5),
        (drift_metrics['of_px_err'], 'orange', 'Optical Flow Tracker', 2.5, 1.0),
        (drift_metrics['rt_px_err'], 'purple', 'RANSAC Homography Tracker', 2.5, 1.0),
        (drift_metrics_b['classic_px_err'], 'teal', 'LK+RANSAC Tracker', 2.5, 1.0),
        (drift_metrics_b['naive_px_err'], 'goldenrod', 'SLAM w/o LiDAR gating Tracker', 2.5, 1.0),
        (drift_metrics['vt_px_err'], 'green', 'SLAM+LiDAR gating Tracker (ours)', 3, 1.0),
    ]
    outliers_2d = _plot_series_with_scale_guard(
        ax_2d, time_axis, series_2d, '2D Distance to Target (pixels)',
        '(a) 2D Image Tracking Accuracy - All Methods', 'px')

    series_3d = [
        (drift_metrics['raw_drift'], 'red', 'Unconstrained', 2, 0.5),
        (drift_metrics['of_drift'], 'orange', 'Optical Flow', 2.5, 1.0),
        (drift_metrics['ransac_drift'], 'purple', 'RANSAC Homography', 2.5, 1.0),
        (drift_metrics_b['classic_drift'], 'teal', 'LK+RANSAC', 2.5, 1.0),
        (drift_metrics_b['naive_drift'], 'goldenrod', 'SLAM w/o LiDAR gating', 2.5, 1.0),
        (drift_metrics['stab_drift'], 'green', 'SLAM+LiDAR gating (ours)', 3, 1.0),
    ]
    outliers_3d = _plot_series_with_scale_guard(
        ax_3d, time_axis, series_3d, '3D Error to Target (meters)',
        '(b) 3D Spatial Disruption - All Methods', 'm')
    ax_3d.set_xlabel('Time (seconds)', fontsize=12)
    fig2.tight_layout()

    fig2.savefig(os.path.join(save_dir, 'Figure_2_Tracking_Stability_Merged.pdf'), dpi=300, bbox_inches='tight')

    if outliers_2d or outliers_3d:
        print("\n[FIGURE 2 MERGED] Axis-scale guard triggered (curve peak >"
              f" {OUTLIER_SCALE_FACTOR}x the group median) -- these methods run off-chart"
              " but are annotated with their true peak, not hidden:")
        for lbl, pk in outliers_2d:
            print(f"    2D panel: {lbl} peak = {pk:.1f} px")
        for lbl, pk in outliers_3d:
            print(f"    3D panel: {lbl} peak = {pk:.2f} m")

    df_merged = pd.DataFrame({
        'Time_Seconds': time_axis,
        'Unconstrained_2D_Error_px': drift_metrics['raw_px_err'], 'OpticalFlow_2D_Error_px': drift_metrics['of_px_err'],
        'RANSACHomography_2D_Error_px': drift_metrics['rt_px_err'], 'NaiveSLAMOnly_2D_Error_px': drift_metrics_b['naive_px_err'],
        'ClassicStabilization_2D_Error_px': drift_metrics_b['classic_px_err'], 'VirtualTripod_2D_Error_px': drift_metrics['vt_px_err'],
        'Unconstrained_3D_Error_m': drift_metrics['raw_drift'], 'OpticalFlow_3D_Error_m': drift_metrics['of_drift'],
        'RANSACHomography_3D_Error_m': drift_metrics['ransac_drift'], 'NaiveSLAMOnly_3D_Error_m': drift_metrics_b['naive_drift'],
        'ClassicStabilization_3D_Error_m': drift_metrics_b['classic_drift'], 'VirtualTripod_3D_Error_m': drift_metrics['stab_drift'],
    })
    df_merged.to_csv(os.path.join(save_dir, 'Figure2_Merged_Tracking_Errors_Data.csv'), index=False)
    print(f"\n[FIGURE 2 MERGED] Wrote {os.path.join(save_dir, 'Figure_2_Tracking_Stability_Merged.pdf')}")

    return {
        '2d_panel_px': [{'method': lbl, 'peak': pk} for lbl, pk in outliers_2d],
        '3d_panel_m': [{'method': lbl, 'peak': pk} for lbl, pk in outliers_3d],
    }

def render_merged_fft_spectrum(fft_metrics, fft_metrics_b, fps, save_dir=DATA_DIR):
    freqs = fft_metrics['freqs']

    fig3, ax_fft = plt.subplots(1, 1, figsize=(10, 5))
    fig3.canvas.manager.set_window_title('Figure 3: Vibration Spectrum Extraction')

    ax_fft.plot(freqs, fft_metrics['raw_fft'], color='red', label='Unconstrained EVM', linewidth=2, alpha=0.5)
    ax_fft.plot(freqs, fft_metrics['of_fft'], color='orange', label='Optical Flow EVM', linewidth=2.5)
    ax_fft.plot(freqs, fft_metrics['ransac_fft'], color='purple', label='RANSAC Homography EVM', linewidth=2.5)
    ax_fft.plot(freqs, fft_metrics_b['classic_fft'], color='teal', label='LK+RANSAC EVM', linewidth=2.5)
    ax_fft.plot(freqs, fft_metrics_b['naive_fft'], color='goldenrod', label='SLAM w/o LiDAR gating EVM', linewidth=2.5)
    ax_fft.plot(freqs, fft_metrics['stab_fft'], color='blue', label='SLAM+LiDAR gating EVM', linewidth=2.5)
    ax_fft.axvline(x=TARGET_FREQ_HZ, color='green', linestyle=':', linewidth=2, label=f'Target ({TARGET_FREQ_HZ} Hz)')
    ax_fft.set_title('Vibration Spectrum Extraction Comparison', fontsize=14, fontweight='bold')
    ax_fft.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_fft.set_ylabel('FFT Amplitude', fontsize=12)
    ax_fft.set_xlim(0, fps/2)
    ax_fft.legend(loc='upper right', fontsize=11)
    ax_fft.grid(True, linestyle='--', alpha=0.6)
    fig3.tight_layout()

    fig3.savefig(os.path.join(save_dir, 'Figure_3_FFT_Spectrum_Merged.pdf'), dpi=300, bbox_inches='tight')

    df_merged = pd.DataFrame({
        'Frequency_Hz': freqs,
        'Unconstrained_FFT_Amplitude': fft_metrics['raw_fft'], 'OpticalFlow_FFT_Amplitude': fft_metrics['of_fft'],
        'RANSACHomography_FFT_Amplitude': fft_metrics['ransac_fft'], 'SLAM_wo_LiDAR_gating_FFT_Amplitude': fft_metrics_b['naive_fft'],
        'LKplusRANSAC_FFT_Amplitude': fft_metrics_b['classic_fft'], 'SLAM_with_LiDAR_gating_FFT_Amplitude': fft_metrics['stab_fft'],
    })
    df_merged.to_csv(os.path.join(save_dir, 'Figure3_Merged_FFT_Spectrum_Data.csv'), index=False)
    print(f"\n[FIGURE 3 MERGED] Wrote {os.path.join(save_dir, 'Figure_3_FFT_Spectrum_Merged.pdf')}")

def render_fft_spectrum_annotated(fft_metrics, fft_metrics_b, fps, save_dir=DATA_DIR):
    freqs = fft_metrics['freqs']
    band_mask = np.abs(freqs - TARGET_FREQ_HZ) <= FFT_TARGET_BAND_HALFWIDTH_HZ
    band_idx = np.where(band_mask)[0]
    if len(band_idx) == 0:
        band_idx = np.array([int(np.argmin(np.abs(freqs - TARGET_FREQ_HZ)))])

    series = [
        (fft_metrics['raw_fft'], fft_metrics['raw_px_err'], 'red', 'Unconstrained EVM', 2, 0.5),
        (fft_metrics['of_fft'], fft_metrics['of_px_err'], 'orange', 'Optical Flow EVM', 2.5, 1.0),
        (fft_metrics['ransac_fft'], fft_metrics['rt_px_err'], 'purple', 'RANSAC Homography EVM', 2.5, 1.0),
        (fft_metrics_b['classic_fft'], fft_metrics_b['classic_px_err'], 'teal', 'LK+RANSAC EVM', 2.5, 1.0),
        (fft_metrics_b['naive_fft'], fft_metrics_b['naive_px_err'], 'goldenrod', 'SLAM w/o LiDAR gating EVM', 2.5, 1.0),
        (fft_metrics['stab_fft'], fft_metrics['vt_px_err'], 'blue', 'SLAM+LiDAR gating EVM', 2.5, 1.0),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6.5))
    fig.canvas.manager.set_window_title('Figure 3 (Annotated): Vibration Spectrum, Log Scale + Target-Frequency Peaks')

    ax.axvspan(TARGET_FREQ_HZ - FFT_TARGET_BAND_HALFWIDTH_HZ, TARGET_FREQ_HZ + FFT_TARGET_BAND_HALFWIDTH_HZ,
               color='green', alpha=0.12, zorder=0, label=f'Target Band (±{FFT_TARGET_BAND_HALFWIDTH_HZ} Hz)')

    peak_rows = []
    for y, px_err, color, label, lw, alpha in series:
        y = np.asarray(y, dtype=float)
        y_floor = np.maximum(y, 1e-6)  # log scale can't show zero/negative amplitude
        ax.plot(freqs, y_floor, color=color, label=label, linewidth=lw, alpha=alpha)

        local_peak_idx = band_idx[np.argmax(y[band_idx])]
        peak_freq, peak_amp = freqs[local_peak_idx], max(y[local_peak_idx], 1e-6)
        noise_floor = max(np.median(np.delete(y, band_idx)), 1e-6) if len(band_idx) < len(y) else np.median(y_floor)
        snr_db = 20 * np.log10(peak_amp / noise_floor)

        px_err_valid = np.asarray(px_err, dtype=float)
        px_err_valid = px_err_valid[np.isfinite(px_err_valid)]
        pct_off_target = 100.0 * np.mean(px_err_valid > REGISTRATION_ERROR_FLAG_THRESHOLD_PX) if len(px_err_valid) else 0.0
        unreliable = pct_off_target > REGISTRATION_ERROR_FLAG_FRACTION_PCT
        marker = 'X' if unreliable else 'o'
        ax.scatter([peak_freq], [peak_amp], color=color, s=90 if unreliable else 70, zorder=5,
                   marker=marker, edgecolor='black', linewidth=1.2 if unreliable else 0.9)
        peak_rows.append((snr_db, color, label, peak_amp, pct_off_target, unreliable))

    peak_rows.sort(key=lambda r: -r[0])

    ax.axvline(x=TARGET_FREQ_HZ, color='black', linestyle=':', linewidth=1.5, label=f'Target ({TARGET_FREQ_HZ} Hz)')
    ax.set_yscale('log')
    ax.set_title('Vibration Spectrum Extraction Comparison (Log Scale, Target-Frequency Peaks Annotated)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('FFT Amplitude (log scale)', fontsize=12)
    ax.set_xlim(0, fps / 2)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    fig.subplots_adjust(right=0.72)

    header_y = 0.66
    line_step = 0.062
    fig.text(0.745, header_y, "Peak @ target band, by Tracking-Signal SNR:", fontsize=10, fontweight='bold', va='top')
    for i, (snr_db, color, label, peak_amp, pct_off_target, unreliable) in enumerate(peak_rows):
        short_label = label.split(' EVM')[0]
        y0 = header_y - line_step * (i + 1)
        fig.text(0.745, y0, f"{short_label}", fontsize=9.5, color=color, fontweight='bold', va='top')
        fig.text(0.745, y0 - 0.022, f"  {peak_amp:.0f} amp, {snr_db:.1f} dB", fontsize=8.5, color=color, va='top')
        if unreliable:
            fig.text(0.745, y0 - 0.041, f"  ⚠ UNRELIABLE ({pct_off_target:.0f}% frames off-target, see Fig 2)",
                      fontsize=7.5, color='firebrick', fontweight='bold', va='top')
        else:
            fig.text(0.745, y0 - 0.041, f"  {pct_off_target:.0f}% frames off-target (validated)",
                      fontsize=7.5, color='gray', va='top')

    fig.text(0.745, header_y - line_step * (len(peak_rows) + 1) - 0.01,
              f"Flag threshold: off-target (>{REGISTRATION_ERROR_FLAG_THRESHOLD_PX:.0f}px,"
              f" ½ ROI_BOX_SIZE) on >{REGISTRATION_ERROR_FLAG_FRACTION_PCT:.0f}% of frames"
              f"\nmarker: ● validated  ✖ unreliable",
              fontsize=7, color='dimgray', va='top', style='italic')

    fig.savefig(os.path.join(save_dir, 'Figure_3_FFT_Spectrum_Annotated.pdf'), dpi=300, bbox_inches='tight')
    print(f"\n[FIGURE 3 ANNOTATED] Wrote {os.path.join(save_dir, 'Figure_3_FFT_Spectrum_Annotated.pdf')}")

def render_registration_vs_spectral_summary(fft_metrics, fft_metrics_b, save_dir=DATA_DIR):
    freqs = fft_metrics['freqs']
    band_mask = np.abs(freqs - TARGET_FREQ_HZ) <= FFT_TARGET_BAND_HALFWIDTH_HZ
    band_idx = np.where(band_mask)[0]
    if len(band_idx) == 0:
        band_idx = np.array([int(np.argmin(np.abs(freqs - TARGET_FREQ_HZ)))])

    methods = [
        ('Unconstrained', fft_metrics['raw_fft'], fft_metrics['raw_px_err'], 'red'),
        ('Optical Flow', fft_metrics['of_fft'], fft_metrics['of_px_err'], 'orange'),
        ('RANSAC Homography', fft_metrics['ransac_fft'], fft_metrics['rt_px_err'], 'purple'),
        ('LK+RANSAC', fft_metrics_b['classic_fft'], fft_metrics_b['classic_px_err'], 'teal'),
        ('Fronto-Parallel', fft_metrics_b['naive_fft'], fft_metrics_b['naive_px_err'], 'goldenrod'),
        ('Virtual Tripod (ours)', fft_metrics['stab_fft'], fft_metrics['vt_px_err'], 'blue'),
    ]

    rows = []
    for name, fft, px_err, color in methods:
        fft = np.asarray(fft, dtype=float)
        local_peak_idx = band_idx[np.argmax(fft[band_idx])]
        peak_amp = max(fft[local_peak_idx], 1e-6)
        noise_floor = max(np.median(np.delete(fft, band_idx)), 1e-6) if len(band_idx) < len(fft) else np.median(fft)
        snr_db = 20 * np.log10(peak_amp / noise_floor)
        px_err_valid = np.asarray(px_err, dtype=float)
        px_err_valid = px_err_valid[np.isfinite(px_err_valid)]
        pct_off_target = 100.0 * np.mean(px_err_valid > REGISTRATION_ERROR_FLAG_THRESHOLD_PX) if len(px_err_valid) else 0.0
        mean_err = float(np.mean(px_err_valid)) if len(px_err_valid) else 0.0
        rows.append((name, pct_off_target, snr_db, float(peak_amp), mean_err, color))

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 6.5))
    fig.canvas.manager.set_window_title('Figure 3 (Companion): Registration Accuracy vs Spectral SNR')

    max_x_visible = max(max(r[1] for r in rows) * 1.15, REGISTRATION_ERROR_FLAG_FRACTION_PCT * 1.3)
    y_lo, y_hi = min(r[2] for r in rows) - 2, max(r[2] for r in rows) + 2
    ax.axvspan(0, REGISTRATION_ERROR_FLAG_FRACTION_PCT, color='green', alpha=0.08, zorder=0, label='Registration-validated zone')
    ax.axvspan(REGISTRATION_ERROR_FLAG_FRACTION_PCT, max_x_visible, color='red', alpha=0.06, zorder=0, label='Unreliable zone')
    ax.axvline(REGISTRATION_ERROR_FLAG_FRACTION_PCT, color='gray', linestyle='--', linewidth=1.5)
    x_span = max_x_visible
    y_span = y_hi - y_lo
    jitter_x = 0.012 * x_span
    jitter_y = 0.012 * y_span
    placed = []
    any_collision = False
    for name, pct_off_target, snr_db, peak_amp, mean_err, color in rows:
        unreliable = pct_off_target > REGISTRATION_ERROR_FLAG_FRACTION_PCT
        marker = 'X' if unreliable else 'o'

        collisions = sum(1 for px, py in placed if abs(px - pct_off_target) / x_span < 0.05 and abs(py - snr_db) / y_span < 0.05)
        placed.append((pct_off_target, snr_db))
        if collisions:
            any_collision = True
            # Deterministic offset (not random) so the figure is reproducible run to run.
            plot_x = pct_off_target + jitter_x * collisions
            plot_y = snr_db + jitter_y * collisions
        else:
            plot_x, plot_y = pct_off_target, snr_db

        ax.scatter([plot_x], [plot_y], color=color, s=160, marker=marker, edgecolor='black', linewidth=1.2, zorder=5)
        y_offset = 7 + collisions * 30
        ax.annotate(f"{name}\n(mean {mean_err:.0f}px)", (plot_x, plot_y), xytext=(9, y_offset),
                    textcoords='offset points', fontsize=9, fontweight='bold', color=color,
                    arrowprops=dict(arrowstyle='-', color=color, alpha=0.5, linewidth=0.8) if collisions else None)

    if any_collision:
        fig.text(0.5, -0.02, "Note: markers that would otherwise fully overlap are nudged slightly for visibility;"
                              " labels show the true values.", ha='center', fontsize=7.5, color='dimgray', style='italic')

    ax.set_xlim(0, max_x_visible)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel(f'% of Frames Off-Target (error > {REGISTRATION_ERROR_FLAG_THRESHOLD_PX:.0f}px, Figure 2)', fontsize=11)
    ax.set_ylabel('Target-Frequency Tracking-Signal SNR, dB (Figure 3)', fontsize=11)
    ax.set_title('Registration Accuracy vs. Tracking-Signal Spectral SNR\n(● registration-validated  ✖ exceeds reliability threshold)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    fig.savefig(os.path.join(save_dir, 'Figure_3_Registration_vs_SNR_Summary.pdf'), dpi=300, bbox_inches='tight')

    df = pd.DataFrame([{
        'Method': r[0], 'Pct_Frames_Off_Target': r[1], 'SNR_dB': r[2], 'Peak_Amplitude': r[3],
        'Mean_Registration_Error_px': r[4], 'Registration_Validated': r[1] <= REGISTRATION_ERROR_FLAG_FRACTION_PCT,
    } for r in rows])
    df.to_csv(os.path.join(save_dir, 'Figure3_Registration_vs_SNR_Data.csv'), index=False)
    print(f"\n[FIGURE 3 COMPANION] Wrote {os.path.join(save_dir, 'Figure_3_Registration_vs_SNR_Summary.pdf')}")

# ====================================================================
# PART 4: TECHNICAL LiDAR GATING FIGURE (Updated Fig 4)
# ====================================================================
def generate_technical_gating_figure(pcd_path, T_cam, K, pixel, P_gt_w, gate_threshold=0.04, save_dir=DATA_DIR):
    print(f"\nGenerating Figure 4: 3-Segment LiDAR Gating Analysis (Threshold: ±{gate_threshold}m)...")
    pcd = o3d.io.read_point_cloud(pcd_path)
    # pcd = pcd.voxel_down_sample(voxel_size=0.05)
    points_w = np.asarray(pcd.points)
    
    R_cam = T_cam[:3, :3]
    t_cam = T_cam[:3, 3]
    points_c = (R_cam.T @ (points_w - t_cam).T).T
    
    P_gt_c = R_cam.T @ (P_gt_w - t_cam)
    gt_depth = P_gt_c[2]
    
    ray_c = np.linalg.inv(K) @ np.array([pixel[0], pixel[1], 1.0])
    ray_c = ray_c / np.linalg.norm(ray_c)
    
    t_proj = np.dot(points_c, ray_c)
    dists_to_ray = np.linalg.norm(points_c - np.outer(t_proj, ray_c), axis=1)
    
    frustum_mask = (points_c[:, 2] > 0) & (dists_to_ray < 1.0) 
    filtered_points_c = points_c[frustum_mask]
    depths = filtered_points_c[:, 2]
    
    # 3-Segment Categorical Logic
    accepted_mask = np.abs(depths - gt_depth) <= gate_threshold
    fg_mask = (gt_depth - depths) > gate_threshold
    bg_mask = (depths - gt_depth) > gate_threshold
    
    plt.rcParams.update({'mathtext.fontset': 'cm', 'font.size': 12})
    fig = plt.figure(figsize=(14, 6))
    fig.canvas.manager.set_window_title('Figure 4: 3-Segment LiDAR Gating Analysis')
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(filtered_points_c[fg_mask, 0], filtered_points_c[fg_mask, 2], c='red', s=10, alpha=0.6, label='Foreground (Occlusions)')
    ax1.scatter(filtered_points_c[bg_mask, 0], filtered_points_c[bg_mask, 2], c='dodgerblue', s=10, alpha=0.6, label='Background (Clutter)')
    ax1.scatter(filtered_points_c[accepted_mask, 0], filtered_points_c[accepted_mask, 2], c='forestgreen', s=15, alpha=0.8, label='Accepted Target Manifold')
    
    ax1.scatter(0, 0, c='black', marker='s', s=100, label='Camera Center ($O_c$)')
    ax1.plot([0, ray_c[0]*15], [0, ray_c[2]*15], 'k--', alpha=0.5, label='Pixel Raycast')
    ax1.scatter(P_gt_c[0], gt_depth, c='yellow', marker='*', s=200, edgecolor='black', label=f'GT Target ($Z_{{ref}}={gt_depth:.2f}m$)')
    
    ax1.axhspan(gt_depth - gate_threshold, gt_depth + gate_threshold, color='palegreen', alpha=0.3, label=rf'Acceptance Gate ($\pm {gate_threshold}m$)')
    ax1.set_title('A. Spatial Raycast Analysis (Top-Down X-Z View)', fontweight='bold')
    ax1.set_xlabel('Camera X-Axis (meters)')
    ax1.set_ylabel('Depth / Camera Z-Axis (meters)')
    ax1.set_ylim(max(0, gt_depth - 3), gt_depth + 4) 
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(True, linestyle=':', alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    counts, bins, patches_hist = ax2.hist(depths, bins=50, edgecolor='black', alpha=0.8)
    
    for i, patch in enumerate(patches_hist):
        bin_center = (bins[i] + bins[i+1]) / 2
        if abs(bin_center - gt_depth) <= gate_threshold: patch.set_facecolor('forestgreen')
        elif bin_center < gt_depth - gate_threshold: patch.set_facecolor('red')
        else: patch.set_facecolor('dodgerblue')
            
    ax2.axvline(gt_depth, color='yellow', linestyle='-', linewidth=2, label=f'Reference Depth ($Z_{{ref}}$)')
    ax2.axvline(gt_depth - gate_threshold, color='green', linestyle='--', linewidth=2)
    ax2.axvline(gt_depth + gate_threshold, color='green', linestyle='--', linewidth=2, label='Gate Boundaries')
    ax2.axvspan(gt_depth - gate_threshold, gt_depth + gate_threshold, color='palegreen', alpha=0.3)
    
    eq_text = r"Filter Condition:" + "\n" + r"$|Z_{ray} - Z_{ref}| \leq \Delta Z_{gate}$"
    ax2.text(0.65, 0.85, eq_text, transform=ax2.transAxes, fontsize=14, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax2.set_title('B. Depth Distribution & 3-Segment Filtering Boundaries', fontweight='bold')
    ax2.set_xlabel('Depth (meters)')
    ax2.set_ylabel('Point Density Count')
    ax2.set_xlim(max(0, gt_depth - 3), gt_depth + 4)
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='red', lw=4), Line2D([0], [0], color='forestgreen', lw=4), Line2D([0], [0], color='dodgerblue', lw=4)]
    ax2.legend(custom_lines, ['Foreground (Rejected)', 'Target (Accepted)', 'Background (Rejected)'], loc='upper left', fontsize=10)
    ax2.grid(True, linestyle=':', alpha=0.6)

    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'Figure_4_LiDAR_Gating_Logic.pdf'), dpi=300, bbox_inches='tight')

def generate_multi_method_heatmap_proof(pcd_path, json_path, ref_idx, view_idx, K, pixel, P_gt_w, n_w, d_w, n_c, d_cam,
                                         baseline_pixels, gate_threshold=0.1, save_dir=DATA_DIR):
    print(f"\nGenerating combined Figure 5 (all methods) for Frame {view_idx}...")

    with open(json_path, 'r') as f: data = json.load(f)
    img_path = os.path.join(save_dir, data['frames'][view_idx]['file_path'].replace('./', ''))
    raw_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = raw_img.shape

    # 1. STABILIZATION MATH (Proposed panel)
    T_anch = np.array(data['frames'][ref_idx]['transform_matrix'])
    T_cam = np.array(data['frames'][view_idx]['transform_matrix'])
    H_k = compute_homography(K, T_anch, T_cam, n_c, d_cam)
    try: H_inv = np.linalg.inv(H_k)
    except np.linalg.LinAlgError: H_inv = np.linalg.pinv(H_k)
    stab_img = cv2.warpPerspective(raw_img, H_inv, (w_img, h_img), flags=cv2.INTER_LINEAR)

    # 2. 3D MATH (shared across all panels)
    json_w, json_h = data.get('w', w_img), data.get('h', h_img)
    scale_x, scale_y = w_img / json_w, h_img / json_h
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x; K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x; K_scaled[1, 2] *= scale_y
    dist_coeffs = np.zeros(4, dtype=np.float32)
    if 'k1' in data: dist_coeffs = np.array([data['k1'], data['k2'], data.get('p1', 0), data.get('p2', 0)])

    R_c2w, t_c2w = T_cam[:3, :3], T_cam[:3, 3]
    if np.dot(n_w, t_c2w) + d_w < 0: n_w, d_w = -n_w, -d_w

    pcd = o3d.io.read_point_cloud(pcd_path)
    points_w = np.asarray(pcd.points)
    signed_plane_dist = np.dot(points_w, n_w) + d_w

    R_w2c, t_w2c = R_c2w.T, -R_c2w.T @ t_c2w
    points_c = (R_w2c @ points_w.T).T + t_w2c

    front_mask = points_c[:, 2] > 0.1
    points_w_front = points_w[front_mask]
    signed_plane_dist = signed_plane_dist[front_mask]

    rvec, _ = cv2.Rodrigues(R_w2c)
    points_2d, _ = cv2.projectPoints(points_w_front, rvec, t_w2c, K_scaled, dist_coeffs)
    u_raw, v_raw = points_2d[:, 0, 0], points_2d[:, 0, 1]

    valid_uv = (u_raw >= 0) & (u_raw < w_img) & (v_raw >= 0) & (v_raw < h_img)
    u_raw, v_raw, signed_plane_dist = u_raw[valid_uv], v_raw[valid_uv], signed_plane_dist[valid_uv]

    target_mask = np.abs(signed_plane_dist) <= gate_threshold
    fg_mask = (0 - signed_plane_dist) > gate_threshold
    bg_mask = signed_plane_dist > gate_threshold
    total_points = len(u_raw)

    # 3. WARP 3D POINTS INTO THE STABILIZED FRAME (Proposed panel)
    pts_hom = np.vstack((u_raw, v_raw, np.ones_like(u_raw)))
    pts_stab_hom = H_inv @ pts_hom
    u_stab = pts_stab_hom[0, :] / pts_stab_hom[2, :]
    v_stab = pts_stab_hom[1, :] / pts_stab_hom[2, :]

    box_w, box_h = FIGURE5_ROI_BOX_SIZE, FIGURE5_ROI_BOX_SIZE
    hud_props = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.75, edgecolor='white', linewidth=1.5)

    # 4. LAYOUT: Panel A, one crop panel per baseline, Panel C -- as a grid
    n_baselines = len(baseline_pixels)
    total_panels = n_baselines + 2
    n_cols = 3
    n_rows = int(np.ceil(total_panels / n_cols))

    plt.rcParams.update({'mathtext.fontset': 'cm', 'font.size': 12, 'font.family': 'sans-serif'})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 7 * n_rows))
    fig.canvas.manager.set_window_title(f'Figure 5 (All Methods): 3-Segment Semantic Proof (Frame {view_idx})')
    axes_flat = np.atleast_1d(axes).flatten()
    for ax in axes_flat[total_panels:]:
        ax.axis('off')  # hide any unused grid cells

    # y positions given explicitly (rather than relying on suptitle's default, which sits
    # close enough to 0.97 to collide with the formula box below it -- confirmed visually).
    fig.suptitle('Visualizing Spatial Pruning: 2D Dense Pixel Bounding Boxes vs. 3D Sparse Volumetric Gating',
                 fontsize=18, fontweight='bold', y=0.99)
    formula_text = r"$\bf{Signal\ Purity\ Calculation:}$ " + r"$Purity = \left( \frac{\mathrm{Target\ Plane\ Points\ in\ ROI}}{\mathrm{Total\ Points\ in\ ROI}} \right) \times 100$"
    fig.text(0.5, 0.955, formula_text, ha='center', fontsize=13, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # --- Panel A: Full-Resolution Frame + Sparse LiDAR Mask (shown once) ---
    img_a = raw_img.copy()
    for px, py in zip(u_raw[bg_mask], v_raw[bg_mask]): cv2.circle(img_a, (int(px), int(py)), 2, (30, 144, 255), -1)
    for px, py in zip(u_raw[fg_mask], v_raw[fg_mask]): cv2.circle(img_a, (int(px), int(py)), 2, (255, 50, 50), -1)
    for px, py in zip(u_raw[target_mask], v_raw[target_mask]): cv2.circle(img_a, (int(px), int(py)), 3, (0, 255, 50), -1)

    axA = axes_flat[0]
    axA.imshow(img_a)
    add_zoomed_inset(axA, img_a, pixel[0], pixel[1], FIGURE5_ROI_BOX_SIZE, color='yellow')
    axA.set_title('(a) Full-Resolution Frame + Sparse LiDAR Mask', fontweight='bold', fontsize=13)
    axA.axis('off')
    stat_text_A = f"SPARSE COMPUTE LOAD:\nFull Frame Pixels: {(w_img * h_img):,}\nLiDAR Points Evaluated: {total_points:,}\nSignal Purity: Unknown"
    axA.text(0.03, 0.04, stat_text_A, transform=axA.transAxes, color='white', fontweight='bold', fontsize=10, bbox=hud_props, va='bottom')

    # --- One crop panel per baseline method ---
    panel_idx = 1
    for label, (bu, bv) in baseline_pixels.items():
        raw_u, raw_v = int(bu), int(bv)
        in_box_mask = (u_raw >= raw_u - box_w//2) & (u_raw <= raw_u + box_w//2) & (v_raw >= raw_v - box_h//2) & (v_raw <= raw_v + box_h//2)
        box_total = np.sum(in_box_mask)
        box_target = np.sum(target_mask & in_box_mask)
        box_noise = np.sum((fg_mask | bg_mask) & in_box_mask)
        box_purity = (box_target / box_total) * 100 if box_total > 0 else 0

        img_b = raw_img.copy()
        for px, py in zip(u_raw[bg_mask & in_box_mask], v_raw[bg_mask & in_box_mask]): cv2.circle(img_b, (int(px), int(py)), 4, (30, 144, 255), -1)
        for px, py in zip(u_raw[fg_mask & in_box_mask], v_raw[fg_mask & in_box_mask]): cv2.circle(img_b, (int(px), int(py)), 4, (255, 50, 50), -1)
        for px, py in zip(u_raw[target_mask & in_box_mask], v_raw[target_mask & in_box_mask]): cv2.circle(img_b, (int(px), int(py)), 6, (0, 255, 50), -1)

        ax = axes_flat[panel_idx]
        ax.imshow(img_b)
        add_zoomed_inset(ax, img_b, raw_u, raw_v, FIGURE5_ROI_BOX_SIZE, color='orange')
        rect2d = patches.Rectangle((raw_u - box_w//2, raw_v - box_h//2), box_w, box_h, linewidth=3, edgecolor='orange', facecolor='none', linestyle='--')
        ax.add_patch(rect2d)
        ax.set_title(f'(b{panel_idx}) 2D Vision Crop ({label})', fontweight='bold', fontsize=13)
        ax.axis('off')
        stat_text_B = f"ROI BOTTLENECK:\nTrapped LiDAR Points: {box_total:,}\nNoise (Red/Blue): {box_noise:,} pts\nSignal Purity: {box_purity:.1f}%"
        ax.text(0.03, 0.04, stat_text_B, transform=ax.transAxes, color='white', fontweight='bold', fontsize=10, bbox=hud_props, va='bottom')
        panel_idx += 1

    # --- Panel C: Proposed 3D Planar Manifold Gate (shown once) ---
    gate_target = np.sum(target_mask)

    dark_img = (stab_img * 0.3).astype(np.uint8)
    for px, py in zip(u_stab[target_mask], v_stab[target_mask]):
        cv2.circle(dark_img, (int(px), int(py)), 6, (0, 255, 50), -1)

    axC = axes_flat[total_panels - 1]
    axC.imshow(dark_img)
    add_zoomed_inset(axC, dark_img, pixel[0], pixel[1], FIGURE5_ROI_BOX_SIZE, color='#00FF32')
    rect3d = patches.Rectangle((pixel[0] - box_w//2, pixel[1] - box_h//2), box_w, box_h, linewidth=3, edgecolor='#00FF32', facecolor='none', linestyle='-')
    axC.add_patch(rect3d)
    axC.set_title('(c) Proposed 3D Planar Manifold Gate', fontweight='bold', fontsize=13)
    axC.axis('off')
    stat_text_C = f"PROPOSED GATE:\nAccepted Target Points: {gate_target:,}\nNoise Passed: 0 pts\nSignal Purity: 100.0%"
    axC.text(0.03, 0.04, stat_text_C, transform=axC.transAxes, color='#00FF32', fontweight='bold', fontsize=10, bbox=hud_props, va='bottom')

    fig.text(0.5, 0.005, "Red = Foreground  |  Green = Target  |  Blue = Background", ha='center', fontweight='bold', color='black',
              bbox=dict(facecolor='white', alpha=0.9, edgecolor='black', boxstyle='round,pad=0.4'))

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    save_path = os.path.join(save_dir, f'Figure_5_Semantic_Proof_Frame_{view_idx}_AllMethods.pdf')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[FIGURE 5 ALL METHODS] Wrote {save_path}")


def project_roi_to_3d(pcd_path, K, frame_metadata, roi_pixel, n_w, d_w, gate_threshold=0.04):
    print("\n--- PROJECTING STATIC 3D FRUSTUM ---")
    pcd = o3d.io.read_point_cloud(pcd_path)
    points_w = np.asarray(pcd.points)
    
    c2w_matrix = np.array(frame_metadata['transform_matrix'])
    R_w2c = c2w_matrix[:3, :3].T
    T_c2w = c2w_matrix[:3, 3]
    
    # 1. Project to Camera Coordinates
    points_c = (R_w2c @ (points_w - T_c2w).T).T
    front_mask = points_c[:, 2] > 0.1
    points_c = points_c[front_mask]
    points_w = points_w[front_mask]
    
    # 2. Apply EXACT Planar Depth Logic (from Figure 5)
    signed_plane_dist = np.dot(points_w, n_w) + d_w
    if np.dot(n_w, T_c2w) + d_w < 0:
        signed_plane_dist = -signed_plane_dist
        
    target_mask = np.abs(signed_plane_dist) <= gate_threshold
    
    # 3. Apply EXACT 2D Bounding Box Size (ROI_BOX_SIZE = 250)
    u = (K[0, 0] * points_c[:, 0] / points_c[:, 2]) + K[0, 2]
    v = (K[1, 1] * points_c[:, 1] / points_c[:, 2]) + K[1, 2]
    
    box_w, box_h = ROI_BOX_SIZE, ROI_BOX_SIZE
    in_box_mask = (np.abs(u - roi_pixel[0]) <= box_w // 2) & (np.abs(v - roi_pixel[1]) <= box_h // 2)
    
    # 4. Combine Masks
    final_mask = target_mask & in_box_mask
    patch_pts = points_w[final_mask]
    
    if len(patch_pts) == 0: 
        return pcd, [], c2w_matrix
        
    return pcd, patch_pts, c2w_matrix

# ====================================================================
# PART 7: 3D FRUSTUM PATCHING & DETAILED GUI DASHBOARD
# ====================================================================
def visualize_3d_patch(pcd, patch_pts, c2w_matrix):
    print("\nLaunching Rviz-Style Open3D GUI Dashboard...")
    
    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window("3D Defect Localization [Rviz Mode]", 1440, 900)
    em = window.theme.font_size
    
    # Setup Scene
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    scene.scene.set_background([0.05, 0.05, 0.07, 1.0]) # Dark Rviz-style background
    
    # --- 1. THICK XYZ AXES ---
    # Red = X, Green = Y, Blue = Z
    thick_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0, origin=[0, 0, 0])
    mat_axes = rendering.MaterialRecord()
    mat_axes.shader = "defaultLit"
    scene.scene.add_geometry("global_thick_axes", thick_axes, mat_axes)
    
    # --- 2. RVIZ-STYLE GLOBAL MAP (Height Gradient) ---
    points = np.asarray(pcd.points)
    z_values = points[:, 2]
    z_norm = (z_values - z_values.min()) / (z_values.max() - z_values.min() + 1e-6)
    
    colors = np.zeros((len(z_norm), 3))
    colors[:, 0] = 0.2 + z_norm * 0.2 # R
    colors[:, 1] = 0.2 + z_norm * 0.2 # G
    colors[:, 2] = 0.3 + z_norm * 0.3 # B
    pcd.colors = o3d.utility.Vector3dVector(colors)

    mat_base = rendering.MaterialRecord()
    mat_base.shader = "defaultUnlit"
    mat_base.point_size = 3.0 # Solid wall appearance
    scene.scene.add_geometry("global_map", pcd, mat_base)
    
    # --- 3. DEFECT HIGHLIGHTING & SIDEBAR ---
    sidebar = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
    sidebar.add_child(gui.Label("STRUCTURAL ANALYSIS REPORT"))
    sidebar.add_child(gui.Label("-" * 30))

    # Save original camera parameters for the Reset button
    initial_eye = c2w_matrix[:3, 3]
    initial_center = [0, 0, 0]
    initial_up = [0, 0, 1]

    if len(patch_pts) > 0:
        mat_roi = rendering.MaterialRecord()
        mat_roi.shader = "defaultUnlit"
        mat_roi.point_size = 12.0 # Significantly larger than map points
        
        pcd_roi = o3d.geometry.PointCloud()
        pcd_roi.points = o3d.utility.Vector3dVector(patch_pts)
        pcd_roi.paint_uniform_color([1.0, 0.0, 0.0]) # Pure "Warning" Red
        scene.scene.add_geometry("defect_target", pcd_roi, mat_roi)
        
        # Bounding Box (Marker Array Style)
        aabb = pcd_roi.get_axis_aligned_bounding_box()
        bbox_lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
        bbox_lines.paint_uniform_color([1.0, 0.2, 0.2])
        
        mat_box = rendering.MaterialRecord()
        mat_box.shader = "unlitLine"
        mat_box.line_width = 5.0
        scene.scene.add_geometry("target_bbox", bbox_lines, mat_box)
        
        # Centroid Math for Sidebar
        centroid = np.mean(patch_pts, axis=0)
        extents = aabb.get_extent()
        
        # Add labels to Sidebar
        sidebar.add_child(gui.Label(f"Status: ANOMALY DETECTED"))
        sidebar.add_child(gui.Label(f"Point Count: {len(patch_pts)}"))
        sidebar.add_child(gui.Label(f"Location [X,Y,Z]:"))
        sidebar.add_child(gui.Label(f" [{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"))
        sidebar.add_child(gui.Label(f"Dimensions [W,H,D]:"))
        sidebar.add_child(gui.Label(f" [{extents[0]:.2f}m, {extents[1]:.2f}m, {extents[2]:.2f}m]"))

        # Look at the defect and save center
        initial_center = centroid
        scene.look_at(initial_center, initial_eye, initial_up)
    else:
        sidebar.add_child(gui.Label("Status: NO DEFECTS FOUND"))
        scene.setup_camera(60, scene.scene.bounding_box, initial_eye)
        initial_center = scene.scene.bounding_box.get_center()

    # --- 4. RESET POV BUTTON ---
    sidebar.add_child(gui.Label(" ")) # Spacer
    reset_btn = gui.Button("Reset Original POV")
    reset_btn.horizontal_padding_em = 1.0
    reset_btn.vertical_padding_em = 0.5
    sidebar.add_child(reset_btn)

    # Callback function for button click
    def on_reset_clicked():
        if len(patch_pts) > 0:
            scene.look_at(initial_center, initial_eye, initial_up)
        else:
            scene.setup_camera(60, scene.scene.bounding_box, initial_eye)
            
    reset_btn.set_on_clicked(on_reset_clicked)

    # --- 5. LAYOUT LOGIC ---
    def on_layout(layout_context):
        r = window.content_rect
        side_width = 300
        scene.frame = gui.Rect(r.x, r.y, r.width - side_width, r.height)
        sidebar.frame = gui.Rect(r.x + r.width - side_width, r.y, side_width, r.height)

    window.set_on_layout(on_layout)
    window.add_child(scene)
    window.add_child(sidebar)
    gui.Application.instance.run()
# ====================================================================
# MAIN EXECUTION PIPELINE
# ====================================================================
def file_md5(path, chunk_size=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def write_run_manifest(save_dir, ref_idx, num_frames, P_gt, extra=None):
    manifest = {
        'timestamp': datetime.datetime.now().isoformat(),
        'json_file': JSON_FILE,
        'json_md5': file_md5(JSON_FILE),
        'pcd_file': PCD_FILE,
        'pcd_md5': file_md5(PCD_FILE),
        'ref_idx': ref_idx,
        'num_frames': num_frames,
        'crosshair_pixel': list(CROSSHAIR_PIXEL),
        'random_seed': RANDOM_SEED,
        'P_gt_world': None if P_gt is None else [float(P_gt[0]), float(P_gt[1]), float(P_gt[2])],
    }
    if extra:
        manifest.update(extra)
    path = os.path.join(save_dir, 'run_manifest.json')
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[PROVENANCE] Wrote {path}")
    print(f"[PROVENANCE] ref_idx={ref_idx} / num_frames={num_frames} / P_gt={manifest['P_gt_world']}")
    print(f"[PROVENANCE] json_md5={manifest['json_md5'][:12]}... pcd_md5={manifest['pcd_md5'][:12]}...")
    return manifest

def write_ransac_diagnostics(save_dir, ransac_inlier_counts, ransac_failed, ref_idx):
    df = pd.DataFrame({
        'Frame_Index': np.arange(len(ransac_inlier_counts)),
        'Inlier_Count': ransac_inlier_counts,
        'Held_Last_Valid_H': ransac_failed,
    })
    path = os.path.join(save_dir, 'RANSAC_Homography_Diagnostics.csv')
    df.to_csv(path, index=False)

    non_anchor = [c for i, c in enumerate(ransac_inlier_counts) if i != ref_idx]
    n_failed = sum(1 for i, f in enumerate(ransac_failed) if i != ref_idx and f)
    n_total = len(non_anchor)
    summary = {
        'diagnostics_csv': path,
        'mean_inliers': float(np.mean(non_anchor)) if non_anchor else None,
        'median_inliers': float(np.median(non_anchor)) if non_anchor else None,
        'frames_below_min_inliers': n_failed,
        'frames_total': n_total,
        'failure_rate_pct': round(100.0 * n_failed / n_total, 2) if n_total else None,
        'min_inliers_threshold': RANSAC_MIN_INLIERS,
        'ransac_reproj_threshold_px': RANSAC_REPROJ_THRESHOLD,
    }
    print(f"[RANSAC BASELINE] mean inliers={summary['mean_inliers']:.1f}, "
          f"{n_failed}/{n_total} frames ({summary['failure_rate_pct']}%) held last valid H "
          f"(< {RANSAC_MIN_INLIERS} inliers).")
    return summary

if __name__ == "__main__":
    o3d.utility.random.seed(RANDOM_SEED)

    K, frames = load_slam_data(JSON_FILE)

    print("\n" + "="*50)
    print(f"Dataset Loaded: {len(frames)} total frames found.")

    ref_idx = REF_IDX
    if ref_idx < 0 or ref_idx >= len(frames):
        raise ValueError(
            f"REF_IDX={ref_idx} is out of range for this dataset ({len(frames)} frames). "
            f"This transforms.json does not match the run REF_IDX was recorded against - "
            f"do not silently clamp/re-derive it, pick a value deliberately and update REF_IDX."
        )
    print(f"Using anchor frame REF_IDX = {ref_idx}")
    print("="*50 + "\n")

    # 1. Initialize Anchor State
    T_anch = np.array(frames[ref_idx]['transform_matrix'])

    # 2. Extract Plane utilizing the hardcoded CROSSHAIR_PIXEL
    n_c, d, P_gt, n_w, d_w = extract_structural_plane(PCD_FILE, T_anch, K, CROSSHAIR_PIXEL)

    write_run_manifest(DATA_DIR, ref_idx, len(frames), P_gt)

    # 3. Build Tracking Vectors & Math (raw / optical-flow / RANSAC-homography baseline / virtual tripod)
    (raw_t, stab_t, ransac_t, of_sig, frames_meta, plot_data, raw_pixels,
     raw_3d, of_3d, ransac_3d, stab_3d,
     raw_px_err, of_px_err, rt_px_err, vt_px_err, of_coords,
     ransac_inlier_counts, ransac_failed, ransac_homogs) = build_master_pipeline(
        JSON_FILE, DATA_DIR, DOWNSCALE_FACTOR, ref_idx, n_c, d, n_w, d_w, P_gt
    )

    ransac_summary = write_ransac_diagnostics(DATA_DIR, ransac_inlier_counts, ransac_failed, ref_idx)
    print("[RANSAC BASELINE] NOTE: this method has no native depth/3D capability - its 3D")
    print("                  error is computed by re-projecting its tracked 2D pixel through")
    print("                  OUR SLAM poses + LiDAR plane for evaluation purposes only.")

    # 4. Render Core Matplotlib Figures
    drift_metrics = render_plots(raw_t, stab_t, ransac_t, of_sig, plot_data, frames_meta, ref_idx, P_gt,
                                  raw_pixels, raw_3d, of_3d, ransac_3d, stab_3d,
                                  raw_px_err, of_px_err, rt_px_err, vt_px_err, FPS, DOWNSCALE_FACTOR)

    def _final_valid(series):
        vals = [v for v in series if v is not None and not np.isnan(v)]
        return vals[-1] if vals else None

    ransac_summary.update({
        'mean_3d_error_m': float(np.nanmean(drift_metrics['ransac_drift'])),
        'final_3d_error_m': _final_valid(drift_metrics['ransac_drift']),
        'mean_2d_error_px': float(np.nanmean(drift_metrics['rt_px_err'])),
    })
    write_run_manifest(DATA_DIR, ref_idx, len(frames), P_gt, extra={'ransac_homography_baseline': ransac_summary})
    (naive_t, classic_t, stab_t_b, frames_meta_b, plot_data_b,
     naive_3d, classic_3d, stab_3d_b,
     naive_px_err, classic_px_err, vt_px_err_b, classic_homogs) = build_ablation_pipeline_b(
        JSON_FILE, DATA_DIR, DOWNSCALE_FACTOR, ref_idx, n_c, d, n_w, d_w, P_gt
    )
    print("[FIGURE SET B] NOTE: Naive SLAM-only Plane and Classic Stabilization have no")
    print("               native depth/3D capability - their 3D error is computed by")
    print("               re-projecting their tracked 2D pixel through OUR SLAM poses +")
    print("               LiDAR plane for evaluation purposes only.")
    drift_metrics_b = render_plots_b(plot_data, naive_t, classic_t, stab_t_b, plot_data_b, frames_meta_b, ref_idx, P_gt,
                                      naive_3d, classic_3d, stab_3d_b,
                                      naive_px_err, classic_px_err, vt_px_err_b, FPS, DOWNSCALE_FACTOR)

    fig2_scale_report = render_merged_tracking_stability(drift_metrics, drift_metrics_b, FPS)
    render_merged_fft_spectrum(drift_metrics, drift_metrics_b, FPS)
    render_fft_spectrum_annotated(drift_metrics, drift_metrics_b, FPS)
    render_registration_vs_spectral_summary(drift_metrics, drift_metrics_b)

    write_run_manifest(DATA_DIR, ref_idx, len(frames), P_gt, extra={
        'ransac_homography_baseline': ransac_summary,
        'figure_set_b': {
            'naive_plane_assumed_depth_m': NAIVE_PLANE_ASSUMED_DEPTH,
            'classic_stab_smoothing_window_frames': CLASSIC_STAB_SMOOTHING_WINDOW,
            'naive_mean_3d_error_m': float(np.nanmean(drift_metrics_b['naive_drift'])),
            'naive_final_3d_error_m': _final_valid(drift_metrics_b['naive_drift']),
            'classic_mean_3d_error_m': float(np.nanmean(drift_metrics_b['classic_drift'])),
            'classic_final_3d_error_m': _final_valid(drift_metrics_b['classic_drift']),
        },
        'figure_2_merged_offchart_peaks': fig2_scale_report,
    })

    generate_technical_gating_figure(PCD_FILE, T_anch, K, CROSSHAIR_PIXEL, P_gt, gate_threshold=0.04)

    HEATMAP_VIEW_IDX = min(ref_idx + 100, len(frames) - 1)
    assert HEATMAP_VIEW_IDX != ref_idx, "HEATMAP_VIEW_IDX must differ from ref_idx to show any drift"
    T_view = np.array(frames[HEATMAP_VIEW_IDX]['transform_matrix'])
    anch_pixel_hom = np.array([CROSSHAIR_PIXEL[0], CROSSHAIR_PIXEL[1], 1.0])

    def _forward_project(H_fwd):
        u_hom = H_fwd @ anch_pixel_hom
        return (u_hom[0] / u_hom[2], u_hom[1] / u_hom[2])

    baseline_pixels = {
        "Optical Flow Tracker": tuple(of_coords[HEATMAP_VIEW_IDX]),
        "RANSAC Homography": _forward_project(ransac_homogs[HEATMAP_VIEW_IDX]),
        "Fronto-Parallel (no LiDAR)": _forward_project(
            compute_homography(K, T_anch, T_view, np.array([0.0, 0.0, 1.0]), NAIVE_PLANE_ASSUMED_DEPTH)
        ),
        "LK+RANSAC": _forward_project(classic_homogs[HEATMAP_VIEW_IDX]),
    }

    generate_multi_method_heatmap_proof(
        pcd_path=PCD_FILE,
        json_path=JSON_FILE,
        ref_idx=ref_idx,
        view_idx=HEATMAP_VIEW_IDX,
        K=K,
        pixel=CROSSHAIR_PIXEL,
        P_gt_w=P_gt,
        n_w=n_w,
        d_w=d_w,
        n_c=n_c,
        d_cam=d,
        gate_threshold=0.04,
        baseline_pixels=baseline_pixels,
    )

    print("\nDisplaying all plots. Close ALL matplotlib windows to launch the Open3D 3D Frustum Mapping Interface...")
    plt.show() 
    
    pcd, patch_pts, c2w_matrix = project_roi_to_3d(
        PCD_FILE, K, frames_meta[ref_idx], CROSSHAIR_PIXEL, n_w, d_w, gate_threshold=0.04
    )
    visualize_3d_patch(pcd, patch_pts, c2w_matrix)