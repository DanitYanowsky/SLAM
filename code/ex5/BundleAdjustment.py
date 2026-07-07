import sys
import os
import random
import numpy as np
import gtsam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex4'))

from sift_features import read_images
from triangulation import read_cameras, CALIB_PATH
from tracking_database import TrackingDB

EX3_DIR          = os.path.join(os.path.dirname(__file__), '..', 'ex3')
EX4_DIR          = os.path.join(os.path.dirname(__file__), '..', 'ex4')
EX5_DIR          = os.path.dirname(__file__)
EXTRINSICS_PATH  = os.path.join(EX3_DIR, 'extrinsics.npy')
TRACKING_DB_PATH = os.path.join(EX4_DIR, 'tracking_db')
GT_PATH          = os.path.join(EX4_DIR, '..', '..', 'dataset', 'dataset', 'poses', '00.txt')

MIN_TRACK_LENGTH = 15


def build_global_poses(extrinsics_path):
    """
    Compose the relative poses from ex3 into absolute world-to-camera transforms.

    extrinsics[0] = I (frame 0, world frame)
    extrinsics[i] = T_{i-1 -> i}  (cam_{i-1} coords -> cam_i coords)

    Returns list of 4x4 matrices T_wc[i] such that:
        X_cam_i = T_wc[i] @ X_world_h
    """
    rel = np.load(extrinsics_path)   # (N, 4, 4)
    T_global = np.eye(4)
    poses = [T_global.copy()]        # frame 0: identity
    for T_rel in rel[1:]:
        T_global = T_rel @ T_global
        poses.append(T_global.copy())
    return poses


def make_stereo_cal(P0, P1):
    """Build gtsam.Cal3_S2Stereo from the left/right projection matrices."""
    K  = P0[:3, :3]
    fx, fy = K[0, 0], K[1, 1]
    s       = K[0, 1]          # skew (0 for KITTI)
    cx, cy  = K[0, 2], K[1, 2]
    # P1[0,3] = -fx * b  →  b = -P1[0,3] / fx  (positive baseline)
    baseline = -P1[0, 3] / fx
    return gtsam.Cal3_S2Stereo(fx, fy, s, cx, cy, baseline)


def stereo_camera_from_global(T_wc, cal):
    """
    Create a gtsam.StereoCamera for one frame.

    T_wc is the 4x4 world-to-camera transform (output of PnP composition).
    gtsam.StereoCamera / Pose3 wants the *camera-to-world* transform, i.e.
    the pose of the camera in world coordinates:
        R_cw = R_wc^T
        t_cw = -R_wc^T @ t_wc   (camera centre in world)
    """
    R_wc = T_wc[:3, :3]
    t_wc = T_wc[:3,  3]
    R_cw = R_wc.T
    t_cw = -R_wc.T @ t_wc
    pose = gtsam.Pose3(gtsam.Rot3(R_cw), gtsam.Point3(t_cw))
    return gtsam.StereoCamera(pose, cal)


def pick_random_track(db, min_length=MIN_TRACK_LENGTH, seed=42):
    """Return a randomly chosen track id of length >= min_length."""
    rng = random.Random(seed)
    candidates = [t for t in db.all_tracks() if len(db.frames(t)) >= min_length]
    if not candidates:
        raise ValueError(f"No track of length >= {min_length} in the database.")
    return rng.choice(candidates)


# ── 5.1  Triangulation via StereoCamera.backproject() ────────────────────────

def triangulate_from_last_frame(cameras, db, track_id, frame_ids):
    """
    Triangulate the 3-D world point using gtsam.StereoCamera.backproject()
    on the *last* frame of the track that has an associated camera.

    Returns (X_world: gtsam.Point3, triangulation_frame_id: int).
    """
    for fid in reversed(frame_ids):
        if fid in cameras:
            last_fid = fid
            break
    else:
        raise ValueError("No frame in 'cameras' dict for this track.")

    cam  = cameras[last_fid]
    lnk  = db.link(last_fid, track_id)
    meas = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
    X_world = cam.backproject(meas)          # gtsam.Point3 in world coords

    print(f"Triangulated 3-D point from frame {last_fid}: "
          f"({X_world[0]:.3f}, {X_world[1]:.3f}, {X_world[2]:.3f})")
    return X_world, last_fid


# ── 5.2  Reprojection error via StereoCamera.project() ───────────────────────

def compute_reprojection_errors(cameras, db, track_id, frame_ids, X_world):
    """
    Project X_world into every frame that has a camera and compute the L2
    reprojection error for the left and right sub-cameras.

    Returns (valid_frame_ids, errors_left, errors_right).
    """
    valid_fids, errors_left, errors_right = [], [], []

    for fid in frame_ids:
        if fid not in cameras:
            continue
        cam  = cameras[fid]
        lnk  = db.link(fid, track_id)
        proj = cam.project(X_world)          # gtsam.StereoPoint2

        err_l = np.linalg.norm([proj.uL() - lnk.x_left,  proj.v() - lnk.y])
        err_r = np.linalg.norm([proj.uR() - lnk.x_right, proj.v() - lnk.y])

        valid_fids.append(fid)
        errors_left.append(err_l)
        errors_right.append(err_r)

    return valid_fids, errors_left, errors_right


def plot_reprojection_errors(valid_fids, errors_left, errors_right,
                              track_id, triangulation_fid, out_path):
    """Plot L2 reprojection error (pixels) per frame for both cameras."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(valid_fids, errors_left,  'o-', color='steelblue', label='Left camera')
    ax.plot(valid_fids, errors_right, 's-', color='tomato',    label='Right camera')
    ax.axvline(triangulation_fid, color='gray', linestyle='--', linewidth=1,
               label=f'Triangulation frame ({triangulation_fid})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reprojection error (L2, pixels)')
    ax.set_title(f'Reprojection error — Track {track_id}  (length={len(valid_fids)})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reprojection error graph saved to {out_path}")


# ── 5.3  Factor error via GenericStereoFactor3D ───────────────────────────────

def compute_factor_errors(cameras, db, track_id, frame_ids, X_world, cal):
    """
    For each frame, build a gtsam.GenericStereoFactor3D with the observed
    stereo point, insert the camera pose and the 3-D landmark into a Values
    object, and evaluate factor.error(values) = 0.5 * ||residual||^2_Sigma.

    Returns (valid_frame_ids, factor_errors).
    """
    noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    L_KEY = gtsam.symbol('l', 0)

    valid_fids, factor_errs = [], []

    for fid in frame_ids:
        if fid not in cameras:
            continue

        lnk    = db.link(fid, track_id)
        P_KEY  = gtsam.symbol('x', fid)
        meas   = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
        factor = gtsam.GenericStereoFactor3D(meas, noise, P_KEY, L_KEY, cal)

        vals = gtsam.Values()
        vals.insert(P_KEY, cameras[fid].pose())
        vals.insert(L_KEY, X_world)

        factor_errs.append(factor.error(vals))
        valid_fids.append(fid)

    return valid_fids, factor_errs


def plot_factor_errors(valid_fids, factor_errs, track_id, triangulation_fid, out_path):
    """Plot GenericStereoFactor3D error per frame."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(valid_fids, factor_errs, 'o-', color='darkorange', label='Factor error')
    ax.axvline(triangulation_fid, color='gray', linestyle='--', linewidth=1,
               label=f'Triangulation frame ({triangulation_fid})')
    ax.set_xlabel('Frame')
    ax.set_ylabel(r'Factor error  ($\frac{1}{2}\|residual\|^2_\Sigma$)')
    ax.set_title(f'Factor error (GenericStereoFactor3D) — Track {track_id}  '
                 f'(length={len(valid_fids)})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Factor error graph saved to {out_path}")


BUNDLE_SIZE   = 10
BA_ITERATIONS = 100


def build_bundle_window(window_fids, db, global_poses, cal, anchored_poses=None):
    """
    Build initial Values for one BA window of frame IDs.

    - One Pose3 per frame  (key 'x', fid)  → 6 DOF each
      If anchored_poses supplies a fid, that optimized Pose3 is used instead of
      the raw odometry pose (used for the shared frame from the previous bundle).
    - One Point3 per track visible in this window (key 'l', tid), triangulated
      from the last valid frame inside the window via backproject().

    Returns (ba_cameras, initial_values, n_landmarks).
    """
    n_poses = len(global_poses)
    valid_fids = [fid for fid in window_fids if fid < n_poses]

    ba_cameras = {fid: stereo_camera_from_global(global_poses[fid], cal)
                  for fid in valid_fids}

    initial_values = gtsam.Values()

    for fid, cam in ba_cameras.items():
        if anchored_poses and fid in anchored_poses:
            initial_values.insert(gtsam.symbol('x', fid), anchored_poses[fid])
        else:
            initial_values.insert(gtsam.symbol('x', fid), cam.pose())

    # Only keep tracks that appear in >= 2 frames inside this window
    window_set = set(valid_fids)
    seen_tracks = set()
    for fid in valid_fids:
        for tid in db.tracks(fid):
            frames_in_window = [f for f in db.frames(tid) if f in window_set]
            if len(frames_in_window) >= 2:
                seen_tracks.add(tid)

    n_landmarks = 0
    for tid in seen_tracks:
        for fid in sorted(db.frames(tid)):   # earliest frame first
            if fid not in ba_cameras:
                continue
            lnk = db.link(fid, tid)

            # Filter 1: positive disparity (point must be in front of camera)
            if lnk.x_left - lnk.x_right < 1.0:
                continue   # try next frame

            meas = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
            X    = ba_cameras[fid].backproject(meas)

            # Filter 2: depth in triangulation camera must be sane
            depth = ba_cameras[fid].pose().transformTo(gtsam.Point3(X))[2]
            if not (0 < depth < 200):
                continue   # try next frame

            # Filter 3: reprojection error < 50 px in EVERY other window frame
            consistent = True
            for other_fid in sorted(db.frames(tid)):
                if other_fid not in ba_cameras or other_fid == fid:
                    continue
                try:
                    other_lnk = db.link(other_fid, tid)
                    proj = ba_cameras[other_fid].project(gtsam.Point3(X))
                    err  = np.linalg.norm([proj.uL() - other_lnk.x_left,
                                           proj.v()  - other_lnk.y])
                    if err > 50.0:
                        consistent = False
                        break
                except Exception:
                    consistent = False
                    break

            if not consistent:
                continue   # try next frame for this track

            initial_values.insert(gtsam.symbol('l', tid), X)
            n_landmarks += 1
            break   # landmark accepted, move to next track

    return ba_cameras, initial_values, n_landmarks


def build_factor_graph(window_fids, ba_cameras, initial_values, db, cal):
    """
    Create a NonlinearFactorGraph with one GenericStereoFactor3D per observation
    in the window, plus a tight prior on the first frame to fix gauge freedom.
    """
    noise       = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)
    graph       = gtsam.NonlinearFactorGraph()

    # Fix the first frame (anchor) — uses whatever pose is in initial_values,
    # so if it came from the previous bundle's result it stays anchored there.
    anchor_key = gtsam.symbol('x', window_fids[0])
    graph.addPriorPose3(anchor_key,
                        initial_values.atPose3(anchor_key),
                        prior_noise)

    for fid in window_fids:
        if fid not in ba_cameras:
            continue
        P_KEY = gtsam.symbol('x', fid)
        for tid in db.tracks(fid):
            L_KEY = gtsam.symbol('l', tid)
            if not initial_values.exists(L_KEY):
                continue
            lnk  = db.link(fid, tid)
            disp = lnk.x_left - lnk.x_right
            if not (1.0 <= disp <= 300.0):
                continue
            meas = gtsam.StereoPoint2(lnk.x_left, lnk.x_right, lnk.y)
            graph.add(gtsam.GenericStereoFactor3D(meas, noise, P_KEY, L_KEY, cal))

    return graph


def optimize_bundle_window(window_fids, ba_cameras, initial_values, db, cal, n_iterations):
    """
    Build the factor graph for this window and run Levenberg-Marquardt for
    exactly n_iterations. Returns (result, graph, error_before, error_after).
    """
    graph = build_factor_graph(window_fids, ba_cameras, initial_values, db, cal)

    # Drop variables that lost all observations after disparity filtering
    graph_keys = set()
    for i in range(graph.size()):
        for key in graph.at(i).keys():
            graph_keys.add(key)
    clean_values = gtsam.Values()
    for key in initial_values.keys():
        if key in graph_keys:
            ch = gtsam.symbolChr(key)
            if ch == ord('x'):
                clean_values.insert(key, initial_values.atPose3(key))
            else:
                clean_values.insert(key, initial_values.atPoint3(key))
    initial_values = clean_values

    error_before = graph.error(initial_values)

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(n_iterations)
    params.setRelativeErrorTol(1e-8)
    params.setAbsoluteErrorTol(1e-8)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    result = optimizer.optimize()

    error_after = graph.error(result)
    return result, graph, error_before, error_after


def _analyse_projection(c, q, lnk, values, cal, factor, label, out_prefix):
    """
    Given Values, project landmark q into frame c.
    Print error + pixel distances; save a side-by-side left/right image.
    """
    pose       = values.atPose3(gtsam.symbol('x', c))
    X_q        = values.atPoint3(gtsam.symbol('l', q))
    stereo_cam = gtsam.StereoCamera(pose, cal)
    proj       = stereo_cam.project(X_q)

    dist_l = np.linalg.norm([proj.uL() - lnk.x_left,  proj.v() - lnk.y])
    dist_r = np.linalg.norm([proj.uR() - lnk.x_right, proj.v() - lnk.y])

    print(f"\n  [{label}]")
    print(f"    Factor error     : {factor.error(values):.6f}")
    print(f"    Left  proj/meas  : ({proj.uL():.1f}, {proj.v():.1f})  /  "
          f"({lnk.x_left:.1f}, {lnk.y:.1f})   dist={dist_l:.3f} px")
    print(f"    Right proj/meas  : ({proj.uR():.1f}, {proj.v():.1f})  /  "
          f"({lnk.x_right:.1f}, {lnk.y:.1f})   dist={dist_r:.3f} px")

    img_l, img_r = read_images(c)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, img, px, mx, side in [
        (axes[0], img_l, proj.uL(), lnk.x_left,  'Left'),
        (axes[1], img_r, proj.uR(), lnk.x_right, 'Right'),
    ]:
        ax.imshow(img, cmap='gray')
        ax.plot(mx,  lnk.y,   'g+', markersize=14, markeredgewidth=2, label='Measurement')
        ax.plot(px,  proj.v(), 'rx', markersize=14, markeredgewidth=2, label='Projection')
        ax.set_title(f'{side} image — frame {c}  [{label}]', fontsize=9)
        ax.legend(fontsize=8)
        ax.axis('off')
    fig.suptitle(f'Track {q}  |  {label}  |  '
                 f'dist_L={dist_l:.2f} px   dist_R={dist_r:.2f} px', fontsize=10)
    fig.tight_layout()
    out = f'{out_prefix}.png'
    fig.savefig(out, dpi=150)
    plt.close()
    print(f"    Saved: {out}")


def analyse_bundle_window(window_fids, init_vals, result, graph, db, cal, out_prefix):
    """
    Print bundle statistics and visualise the worst initial-error factor.
    graph must have been built with build_factor_graph (prior at index 0).
    """
    n_factors  = graph.size()
    n_stereo   = n_factors - 1          # subtract the prior factor
    err_before = graph.error(init_vals)
    err_after  = graph.error(result)

    print(f"\n── Bundle Analysis  frames {window_fids[0]}-{window_fids[-1]} ──────────")
    print(f"  Factors in graph         : {n_factors}  ({n_stereo} stereo + 1 prior)")
    print(f"  Total error  before / after : {err_before:.4f}  /  {err_after:.4f}")
    print(f"  Avg factor   before / after : "
          f"{err_before/n_factors:.4f}  /  {err_after/n_factors:.4f}")

    # Find stereo factor with largest initial error (skip prior at index 0)
    worst_err, worst_idx = -1.0, -1
    for i in range(1, n_factors):
        fe = graph.at(i).error(init_vals)
        if fe > worst_err:
            worst_err, worst_idx = fe, i

    worst_factor = graph.at(worst_idx)
    keys = worst_factor.keys()
    c = gtsam.symbolIndex(keys[0])   # frame id
    q = gtsam.symbolIndex(keys[1])   # track / landmark id
    lnk = db.link(c, q)

    print(f"\n  Worst factor: frame c={c}, landmark q={q},  "
          f"initial error={worst_err:.6f}")

    _analyse_projection(c, q, lnk, init_vals, cal, worst_factor,
                        label='Initial',   out_prefix=f'{out_prefix}_initial')
    _analyse_projection(c, q, lnk, result,    cal, worst_factor,
                        label='Optimized', out_prefix=f'{out_prefix}_optimized')


def _load_gt_centers(gt_path, frame_ids):
    """Return {fid: ndarray(3)} — camera centres in world coords from KITTI GT file."""
    with open(gt_path) as f:
        rows = [list(map(float, l.split())) for l in f]
    centers = {}
    for fid in frame_ids:
        if fid < len(rows):
            T = np.array(rows[fid]).reshape(3, 4)   # camera-to-world [R|t]
            centers[fid] = T[:, 3]                  # camera centre = translation column
    return centers


def _values_cam_centers(values, frame_ids):
    """Return {fid: ndarray(3)} — camera centres extracted from a gtsam.Values."""
    centers = {}
    for fid in frame_ids:
        key = gtsam.symbol('x', fid)
        if values.exists(key):
            centers[fid] = np.array(values.atPose3(key).translation())
    return centers


def plot_full_trajectory_comparison(all_results, global_poses, gt_path, out_path):
    """
    Top-down (X-Z) plot of every optimised camera centre across ALL bundles.
      - orange  : odometry (global_poses, world-to-camera)
      - blue    : after BA (last bundle that contains each frame)
      - green   : ground truth
    """
    n_poses = len(global_poses)

    # Collect all frame ids that appear in at least one bundle result
    all_fids = sorted({fid for fids, _ in all_results for fid in fids if fid < n_poses})

    # Before: camera centre from world-to-camera matrix  c = -R^T t
    before = {}
    for fid in all_fids:
        T = global_poses[fid]
        R, t = T[:3, :3], T[:3, 3]
        before[fid] = -R.T @ t

    # After: use the LAST bundle result that contains each frame (most refined)
    after = {}
    for fids, result in all_results:
        centers = _values_cam_centers(result, fids)
        after.update(centers)   # later bundles overwrite earlier ones for shared frames

    # Ground truth
    gt = _load_gt_centers(gt_path, all_fids)

    fig, ax = plt.subplots(figsize=(14, 10))
    for centers, label, color, marker in [
        (before, 'Odometry (before BA)', 'orange',    'o'),
        (after,  'After BA',             'steelblue', 's'),
        (gt,     'Ground truth',         'green',     '^'),
    ]:
        fids = sorted(centers)
        xs = [centers[f][0] for f in fids]
        zs = [centers[f][2] for f in fids]
        ax.plot(xs, zs, color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(xs, zs, c=color, s=20, marker=marker, zorder=5, label=label)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z — depth (m)')
    ax.set_title('Full trajectory — odometry / after BA / GT (top-down)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Full trajectory comparison saved to {out_path}")


def plot_cameras_comparison(window_fids, init_vals, result, gt_path, out_path):
    """
    Top-down (X-Z) plot of camera centres for the given window:
      - orange  : before optimisation (initial odometry)
      - blue    : after optimisation
      - green   : ground truth
    """
    before  = _values_cam_centers(init_vals, window_fids)
    after   = _values_cam_centers(result,    window_fids)
    gt      = _load_gt_centers(gt_path,      window_fids)

    fig, ax = plt.subplots(figsize=(9, 7))

    for centers, label, color, marker in [
        (before, 'Before optimisation', 'orange',    'o'),
        (after,  'After optimisation',  'steelblue', 's'),
        (gt,     'Ground truth',        'green',     '^'),
    ]:
        fids = sorted(centers)
        xs = [centers[f][0] for f in fids]
        zs = [centers[f][2] for f in fids]
        ax.plot(xs, zs, color=color, linewidth=1.5, alpha=0.7)
        ax.scatter(xs, zs, c=color, s=70, marker=marker, zorder=5, label=label)
        for f, x, z in zip(fids, xs, zs):
            ax.annotate(str(f), (x, z), fontsize=6, ha='center', va='bottom', color=color)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z — depth (m)')
    ax.set_title(f'Camera positions — frames {window_fids[0]}–{window_fids[-1]}  '
                 f'(before / after BA / GT)')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Camera comparison plot saved to {out_path}")


def plot_bundle_result(window_fids, result, out_prefix):
    """
    Visualise the optimised first bundle:
      1. 3D view — camera trajectory via gtsam.utils.plot.plot_trajectory + landmark cloud
      2. Top-down (X-Z) 2D view — cameras + landmarks
    """
    import gtsam.utils.plot as gtsam_plot
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D projection)

    # Collect camera centres (sorted by frame id) and landmark positions
    cam_items = []   # [(fid, ndarray(3))]
    land_pts  = []   # [ndarray(3)]

    for key in result.keys():
        ch = gtsam.symbolChr(key)
        if ch == ord('x'):
            fid = gtsam.symbolIndex(key)
            t   = result.atPose3(key).translation()
            cam_items.append((fid, np.array(t)))
        elif ch == ord('l'):
            land_pts.append(np.array(result.atPoint3(key)))

    cam_items.sort(key=lambda x: x[0])
    cam_pts  = np.array([t for _, t in cam_items]) if cam_items else np.empty((0, 3))
    land_pts = np.array(land_pts)                   if land_pts  else np.empty((0, 3))

    # Remove landmark outliers (must be in front of cameras and within 300 m)
    if len(land_pts):
        ok = ((land_pts[:, 2] > 0) & (land_pts[:, 2] < 300) &
              (np.abs(land_pts[:, 0]) < 150) & (np.abs(land_pts[:, 1]) < 50))
        land_pts = land_pts[ok]

    # ── 3D: gtsam trajectory + matplotlib landmark cloud ─────────────────────
    FIGNUM = 99
    gtsam_plot.plot_trajectory(FIGNUM, result, scale=1.0,
                                title=f'Bundle 3D — frames {window_fids[0]}–{window_fids[-1]}')
    fig3d = plt.figure(FIGNUM)
    ax3d  = fig3d.axes[0]
    if len(land_pts):
        ax3d.scatter(land_pts[:, 0], land_pts[:, 1], land_pts[:, 2],
                     c='tomato', s=1, alpha=0.3, label='Landmarks')
        ax3d.legend(fontsize=7)
    ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
    fig3d.tight_layout()
    out3d = f'{out_prefix}_3d.png'
    fig3d.savefig(out3d, dpi=150)
    plt.close(fig3d)
    print(f"3D bundle plot saved to {out3d}")

    # ── 2D top-down (X–Z) ─────────────────────────────────────────────────────
    fig2d, ax2d = plt.subplots(figsize=(10, 8))
    if len(land_pts):
        ax2d.scatter(land_pts[:, 0], land_pts[:, 2], c='tomato', s=2, alpha=0.3,
                     label=f'Landmarks ({len(land_pts)})')
    if len(cam_pts):
        ax2d.plot(cam_pts[:, 0], cam_pts[:, 2], 'b-', linewidth=1.5, alpha=0.6)
        ax2d.scatter(cam_pts[:, 0], cam_pts[:, 2], c='steelblue', s=70, zorder=5,
                     label=f'Cameras ({len(cam_pts)})')
        for fid, t in cam_items:
            ax2d.annotate(str(fid), (t[0], t[2]), fontsize=6,
                          ha='center', va='bottom', color='navy')
    ax2d.set_xlabel('X (m)')
    ax2d.set_ylabel('Z — depth (m)')
    ax2d.set_title(f'Bundle top-down — frames {window_fids[0]}–{window_fids[-1]}')
    ax2d.legend()
    ax2d.grid(alpha=0.3)
    ax2d.set_aspect('equal')
    fig2d.tight_layout()
    out2d = f'{out_prefix}_topdown.png'
    fig2d.savefig(out2d, dpi=150)
    plt.close(fig2d)
    print(f"Top-down bundle plot saved to {out2d}")


def plot_localization_error(all_results, gt_path, out_path):
    """
    L2 distance (metres) between each optimised keyframe centre and GT,
    plotted as a function of frame index.
    """
    # Build {fid: centre} from last bundle that covers each frame
    est = {}
    for fids, result in all_results:
        est.update(_values_cam_centers(result, fids))

    all_fids = sorted(est)
    gt = _load_gt_centers(gt_path, all_fids)

    fids_common = [f for f in all_fids if f in gt]
    errors = [np.linalg.norm(est[f] - gt[f]) for f in fids_common]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fids_common, errors, color='steelblue', label='l2 localization errors as a function of frames')
    ax.set_xlabel('frames')
    ax.set_ylabel('localization errors from ground truth')
    ax.set_title('keyframe localization error in meters over time')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Localization error plot saved to {out_path}")
    print(f"  Mean localization error: {np.mean(errors):.3f} m  "
          f"Max: {np.max(errors):.3f} m")


def plot_factor_vs_reprojection(errors_left, errors_right, factor_errs,
                                 track_id, out_path):
    """
    Scatter plot: factor error (y) as a function of reprojection error (x).
    Left and right 2-D L2 errors are shown as separate series.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(errors_left,  factor_errs, color='steelblue', alpha=0.8,
               label='Left camera',  marker='o')
    ax.scatter(errors_right, factor_errs, color='tomato',    alpha=0.8,
               label='Right camera', marker='s')
    ax.set_xlabel('Reprojection error (L2, pixels)')
    ax.set_ylabel(r'Factor error  ($\frac{1}{2}\|residual\|^2_\Sigma$)')
    ax.set_title(f'Factor error vs reprojection error — Track {track_id}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Factor vs reprojection graph saved to {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    P0, P1 = read_cameras(CALIB_PATH)

    db = TrackingDB()
    db.load(TRACKING_DB_PATH)

    global_poses = build_global_poses(EXTRINSICS_PATH)
    n_poses = len(global_poses)

    cal = make_stereo_cal(P0, P1)

    track_id  = pick_random_track(db)
    frame_ids = db.frames(track_id)
    print(f"Track {track_id}: length={len(frame_ids)}, frames={frame_ids}")

    cameras = {}
    for fid in frame_ids:
        if fid >= n_poses:
            print(f"  Skip frame {fid}: beyond computed poses ({n_poses} frames)")
            continue
        cameras[fid] = stereo_camera_from_global(global_poses[fid], cal)

    print(f"\nCreated {len(cameras)} gtsam.StereoCamera objects:")
    for fid in sorted(cameras):
        lnk = db.link(fid, track_id)
        t   = cameras[fid].pose().translation()
        print(f"  frame {fid:4d}: centre=({t[0]:+9.4f}, {t[1]:+9.4f}, {t[2]:+9.4f})"
              f"  obs=(xl={lnk.x_left:.1f}, xr={lnk.x_right:.1f}, y={lnk.y:.1f})")

    # ── triangulate from last frame ───────────────────────────────────────────
    X_world, triang_fid = triangulate_from_last_frame(cameras, db, track_id, frame_ids)

    # ── reprojection error plot ───────────────────────────────────────────────
    valid_fids, errors_left, errors_right = compute_reprojection_errors(
        cameras, db, track_id, frame_ids, X_world)

    plot_reprojection_errors(
        valid_fids, errors_left, errors_right, track_id, triang_fid,
        out_path=os.path.join(EX5_DIR, 'out_reprojection_stereo.png'))

    print(f"\nReprojection errors (left ):  mean={np.mean(errors_left):.3f} px  "
          f"max={np.max(errors_left):.3f} px")
    print(f"Reprojection errors (right):  mean={np.mean(errors_right):.3f} px  "
          f"max={np.max(errors_right):.3f} px")

    # ── factor error plot ─────────────────────────────────────────────────────
    fid_fac, factor_errs = compute_factor_errors(
        cameras, db, track_id, frame_ids, X_world, cal)

    plot_factor_errors(
        fid_fac, factor_errs, track_id, triang_fid,
        out_path=os.path.join(EX5_DIR, 'out_factor_errors.png'))

    print(f"\nFactor errors:  mean={np.mean(factor_errs):.4f}  "
          f"max={np.max(factor_errs):.4f}")

    # ── factor error vs reprojection error ────────────────────────────────────
    plot_factor_vs_reprojection(
        errors_left, errors_right, factor_errs, track_id,
        out_path=os.path.join(EX5_DIR, 'out_factor_vs_reprojection.png'))

    # ── BA: sequential windowed optimization ─────────────────────────────────
    # Windows overlap by 1 frame: [0-9], [9-18], [18-27], …
    # The shared (first) frame of each window is initialized with the optimized
    # pose from the previous bundle and then fixed via a tight prior.
    all_results = []
    prev_result = None
    w_idx       = 0
    start       = 0

    print(f"\nBA sequential optimization  "
          f"(window={BUNDLE_SIZE}, iters={BA_ITERATIONS}):")

    while start < n_poses:
        end         = min(start + BUNDLE_SIZE, n_poses)
        window_fids = list(range(start, end))

        # Anchor: use optimized pose from previous bundle for the shared frame
        anchored = {}
        if prev_result is not None:
            shared_key = gtsam.symbol('x', window_fids[0])
            if prev_result.exists(shared_key):
                anchored[window_fids[0]] = prev_result.atPose3(shared_key)

        ba_cams, init_vals, n_lm = build_bundle_window(
            window_fids, db, global_poses, cal, anchored_poses=anchored)

        if not ba_cams:
            start = end
            continue

        result, graph, err_before, err_after = optimize_bundle_window(
            window_fids, ba_cams, init_vals, db, cal, BA_ITERATIONS)

        n_c = len(ba_cams)
        print(f"  [{w_idx:3d}] frames {window_fids[0]:4d}-{window_fids[-1]:4d} | "
              f"{n_c} cams ({6*n_c} params) | {n_lm} landmarks | "
              f"error {err_before:.2f} -> {err_after:.2f}")

        # Detailed analysis + visualisation for the first bundle only
        if w_idx == 0:
            print("\nCamera centres before vs after:")
            for fid in window_fids:
                key = gtsam.symbol('x', fid)
                b = np.array(init_vals.atPose3(key).translation())
                a = np.array(result.atPose3(key).translation())
                print(f"  frame {fid}: before Z={b[2]:.3f}  after Z={a[2]:.3f}  delta={a[2]-b[2]:+.3f}")

        if w_idx == 0:
            analyse_bundle_window(
                window_fids, init_vals, result, graph, db, cal,
                out_prefix=os.path.join(EX5_DIR, 'out_worst_factor'))
            plot_bundle_result(
                window_fids, result,
                out_prefix=os.path.join(EX5_DIR, 'out_bundle0'))
            plot_cameras_comparison(
                window_fids, init_vals, result, GT_PATH,
                out_path=os.path.join(EX5_DIR, 'out_cameras_comparison.png'))

        prev_result = result
        all_results.append((window_fids, result))
        w_idx += 1


        # Slide forward by BUNDLE_SIZE-1 so the last frame becomes the first
        # of the next window (1-frame overlap = shared anchor)
        start = end - 1
        if end == n_poses:
            break

    print(f"\nDone — {w_idx} bundles optimized.")

    if all_results:
        plot_full_trajectory_comparison(
            all_results, global_poses, GT_PATH,
            out_path=os.path.join(EX5_DIR, 'out_full_trajectory_comparison.png'))
        plot_localization_error(
            all_results, GT_PATH,
            out_path=os.path.join(EX5_DIR, 'out_localization_error.png'))

    return cameras, track_id, db, cal, global_poses, all_results


if __name__ == '__main__':
    main()
