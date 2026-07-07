import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2'))
from sift_features import read_images, run_sift
from triangulation import get_stereo_inliers, read_cameras, CALIB_PATH
from tracking_database import TrackingDB, Link

RATIO = 0.8
NUM_FRAMES = 3500
SERIALIZE_BASE = os.path.join(os.path.dirname(__file__), 'tracking_db')
GT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'dataset', 'poses', '00.txt')


# ── build ────────────────────────────────────────────────────────────────────

def _match_consecutive(des_prev: np.ndarray, des_cur: np.ndarray, ratio: float = RATIO):
    """knnMatch prev→cur; returns (knn_pairs, inlier_bool_list).
    Length of both outputs equals des_prev.shape[0] (one entry per query feature).
    """
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des_prev, des_cur, k=2)
    inliers = []
    for pair in knn:
        if len(pair) == 2:
            m, n = pair
            inliers.append(bool(m.distance < ratio * n.distance))
        else:
            inliers.append(False)
    return knn, inliers


def build_tracking_db(num_frames: int = NUM_FRAMES) -> TrackingDB:
    """Build a TrackingDB over `num_frames` consecutive KITTI stereo pairs."""
    db = TrackingDB()

    # ── frame 0 (no previous frame to match) ────────────────────────────────
    img_l, img_r = read_images(0)
    kp_l, des_l = run_sift(img_l)
    kp_r, des_r = run_sift(img_r)
    stereo_matches = get_stereo_inliers(img_l, kp_l, des_l, img_r, kp_r, des_r)
    features, links = TrackingDB.create_links(des_l, kp_l, kp_r, stereo_matches)

    frame_id = db.add_frame(links, features)
    print(f"Frame {frame_id:4d}: {len(links):4d} stereo features")

    # ── frames 1 … num_frames-1 ─────────────────────────────────────────────
    for i in range(1, num_frames):
        img_l, img_r = read_images(i)
        kp_l, des_l = run_sift(img_l)
        kp_r, des_r = run_sift(img_r)
        stereo_matches = get_stereo_inliers(img_l, kp_l, des_l, img_r, kp_r, des_r)
        cur_features, cur_links = TrackingDB.create_links(des_l, kp_l, kp_r, stereo_matches)

        # Match previous-frame left descriptors → current-frame left descriptors.
        # knn has exactly prev_features.shape[0] entries (one per previous feature),
        # which satisfies the add_frame assertion.
        prev_features = db.last_features()
        knn, inliers = _match_consecutive(prev_features, cur_features)

        frame_id = db.add_frame(cur_links, cur_features, knn, inliers)
        n_tracks = len(db.tracks(frame_id))
        print(f"Frame {frame_id:4d}: {len(cur_links):4d} stereo features  "
              f"{sum(inliers):4d} temporal inliers  {n_tracks:4d} active tracks")

    return db


# ── query demos ──────────────────────────────────────────────────────────────

def demo_queries(db: TrackingDB) -> None:
    print("\n─── Database Summary ───────────────────────────────────────")
    print(f"  Frames : {db.frame_num()}")
    print(f"  Tracks : {db.track_num()}")
    print(f"  Links  : {db.link_num()}")

    # --- tracks(frameId) ---
    sample_fid = db.last_frameId // 2
    track_ids = db.tracks(sample_fid)
    print(f"\n[tracks(frameId={sample_fid})] → {len(track_ids)} TrackIds")
    print(f"  First 5: {track_ids[:5]}")

    if not track_ids:
        return

    # --- frames(trackId) ---
    sample_tid = track_ids[0]
    frame_ids = db.frames(sample_tid)
    print(f"\n[frames(trackId={sample_tid})] → FrameIds: {frame_ids}")

    # --- link(frameId, trackId) → (xl, xr, y) ---
    lnk = db.link(frame_ids[0], sample_tid)
    print(f"\n[link(frameId={frame_ids[0]}, trackId={sample_tid})]")
    print(f"  xl={lnk.x_left:.2f}  xr={lnk.x_right:.2f}  y={lnk.y:.2f}")

    # tracks that span more than one frame (length ≥ 2)
    long_tracks = [t for t in db.all_tracks() if len(db.frames(t)) >= 3]
    print(f"\nTracks of length ≥ 3 : {len(long_tracks)}")
    if long_tracks:
        t = long_tracks[0]
        print(f"  Example — TrackId {t} spans frames {db.frames(t)}")


# ── 4.2 tracking statistics ──────────────────────────────────────────────────

def tracking_statistics(db: TrackingDB) -> None:
    """Print tracking stats, excluding trivial (length-1) tracks."""
    # non-trivial tracks only
    track_lengths = [len(db.frames(t)) for t in db.all_tracks() if len(db.frames(t)) > 1]

    if not track_lengths:
        print("No non-trivial tracks found.")
        return

    lengths = np.array(track_lengths)
    # mean links per frame = total links of non-trivial tracks / number of frames
    mean_frame_links = lengths.sum() / db.frame_num()

    print("\n─── Tracking Statistics (non-trivial tracks only) ──────────")
    print(f"  Total tracks          : {len(lengths)}")
    print(f"  Number of frames      : {db.frame_num()}")
    print(f"  Mean track length     : {lengths.mean():.2f}")
    print(f"  Max  track length     : {lengths.max()}")
    print(f"  Min  track length     : {lengths.min()}")
    print(f"  Mean links per frame  : {mean_frame_links:.2f}")


# ── serialization demos ───────────────────────────────────────────────────────

def demo_serialization(db: TrackingDB, base: str = SERIALIZE_BASE) -> None:
    print("\n─── Serialization ──────────────────────────────────────────")

    # per-frame serialization
    fid = 0
    db.serialize_frame(base, fid)
    features, links = TrackingDB.load_frame(base, fid)
    print(f"  Frame {fid} round-trip: {len(links)} links, features {features.shape}")

    # full DB serialization
    db.serialize(base)

    db2 = TrackingDB()
    db2.load(base)
    assert db2.frame_num() == db.frame_num(), "frame count mismatch after reload"
    assert db2.track_num() == db.track_num(), "track count mismatch after reload"
    assert db2.link_num()  == db.link_num(),  "link count mismatch after reload"
    print(f"  Full DB round-trip: {db2.frame_num()} frames, "
          f"{db2.track_num()} tracks, {db2.link_num()} links — OK")


# ── 4.3 track display ────────────────────────────────────────────────────────

def display_track(db: TrackingDB, min_length: int = 6,
                  out_path: str = 'out_track.png') -> None:
    """
    For a track of length >= min_length, show one row per frame:
      left column  — full left image with the feature marked by a red box
      right column — 20x20 crop centred on the feature
    """
    track_id = next(
        (t for t in db.all_tracks() if len(db.frames(t)) >= min_length), None
    )
    if track_id is None:
        print(f"No track of length >= {min_length} found.")
        return

    frame_ids = db.frames(track_id)
    n = len(frame_ids)
    HALF = 10  # half-size of the 20x20 crop

    # n rows, 2 columns: [full image | crop]
    fig, axes = plt.subplots(n, 2, figsize=(10, n * 2.2),
                             gridspec_kw={'width_ratios': [6, 1]})
    axes = np.array(axes).reshape(n, 2)

    for row, fid in enumerate(frame_ids):
        lnk = db.link(fid, track_id)
        cx, cy = int(round(lnk.x_left)), int(round(lnk.y))

        img_l, _ = read_images(fid)
        H, W = img_l.shape

        # ── left column: full image with red box around feature ──────────────
        ax_full = axes[row, 0]
        ax_full.imshow(img_l, cmap='gray', vmin=0, vmax=255)
        rect = patches.Rectangle(
            (cx - HALF, cy - HALF), 2 * HALF, 2 * HALF,
            linewidth=1.5, edgecolor='red', facecolor='none'
        )
        ax_full.add_patch(rect)
        ax_full.plot(cx, cy, 'r+', markersize=8, markeredgewidth=1.5)
        ax_full.set_title(f'Frame {fid}', fontsize=8)
        ax_full.axis('off')

        # ── right column: 20x20 crop ─────────────────────────────────────────
        y0, y1 = max(0, cy - HALF), min(H, cy + HALF)
        x0, x1 = max(0, cx - HALF), min(W, cx + HALF)
        crop = img_l[y0:y1, x0:x1]

        ax_crop = axes[row, 1]
        ax_crop.imshow(crop, cmap='gray', vmin=0, vmax=255)
        ax_crop.plot(cx - x0, cy - y0, 'r+', markersize=6, markeredgewidth=1.5)
        ax_crop.axis('off')

    axes[0, 0].set_title(f'Left image — Track {track_id} (length={n})', fontsize=9)
    axes[0, 1].set_title('Crop', fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Track display saved to {out_path}")


# ── 4.4 connectivity graph ────────────────────────────────────────────────────

def connectivity_graph(db: TrackingDB,
                       out_path: str = 'out_connectivity.png') -> None:
    """
    For each frame F, count tracks that also appear in frame F+1 (outgoing links).
    Plot this as a bar chart over all frames.
    """
    frame_ids = list(db.all_frames())
    outgoing = []
    for fid in frame_ids[:-1]:
        tracks_here = set(db.tracks(fid))
        tracks_next = set(db.tracks(fid + 1))
        outgoing.append(len(tracks_here & tracks_next))

    mean_out = np.mean(outgoing) if outgoing else 0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_ids[:-1], outgoing, color='steelblue', linewidth=1.2)
    ax.axhline(mean_out, color='green', linewidth=1.5, linestyle='--',
               label=f'Mean = {mean_out:.1f}')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Outgoing tracks')
    ax.set_title('Connectivity: outgoing tracks per frame')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Connectivity graph saved to {out_path}")
    print(f"  Mean outgoing tracks per frame: {mean_out:.1f}")


# ── 4.5 inlier percentage per frame ──────────────────────────────────────────

def inlier_percentage_graph(db: TrackingDB,
                            out_path: str = 'out_inliers.png') -> None:
    """
    For each frame F > 0, compute:
      inliers  = tracks shared between frame F and frame F-1
      total    = total stereo features in frame F
      pct      = inliers / total * 100
    Plot as a continuous line with the mean as a green dashed horizontal line.
    """
    frame_ids = list(db.all_frames())
    pct = []
    for fid in frame_ids[1:]:
        shared = len(set(db.tracks(fid)) & set(db.tracks(fid - 1)))
        total  = db.features(fid).shape[0]
        pct.append(100.0 * shared / total if total > 0 else 0.0)

    mean_pct = np.mean(pct) if pct else 0.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(frame_ids[1:], pct, color='steelblue', linewidth=1.2)
    ax.axhline(mean_pct, color='green', linewidth=1.5, linestyle='--',
               label=f'Mean = {mean_pct:.1f}%')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Inlier percentage (%)')
    ax.set_title('Inlier percentage per frame')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Inlier percentage graph saved to {out_path}")
    print(f"  Mean inlier percentage: {mean_pct:.1f}%")


# ── 4.6 track length histogram ────────────────────────────────────────────────

def track_length_histogram(db: TrackingDB,
                           out_path: str = 'out_track_histogram.png') -> None:
    """Histogram of track lengths, excluding trivial (length-1) tracks."""
    lengths = [len(db.frames(t)) for t in db.all_tracks() if len(db.frames(t)) > 1]
    if not lengths:
        print("No non-trivial tracks to plot.")
        return

    lengths = np.array(lengths)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(lengths, bins=range(2, lengths.max() + 2), color='steelblue',
            edgecolor='white', align='left')
    ax.set_xlabel('Track length (frames)')
    ax.set_ylabel('Number of tracks')
    ax.set_title('Track length histogram (non-trivial tracks)')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Track length histogram saved to {out_path}")


# ── 4.7 reprojection error ───────────────────────────────────────────────────

def _load_gt_poses(gt_path: str):
    """Return list of 3×4 camera-to-world matrices (one per frame)."""
    poses = []
    with open(gt_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            poses.append(np.array(vals).reshape(3, 4))
    return poses


def _world_to_cam_44(T_cw: np.ndarray) -> np.ndarray:
    """
    Given a 3×4 camera-to-world matrix [R | t],
    return the 4×4 world-to-camera matrix [R^T | -R^T t; 0 0 0 1].
    """
    R = T_cw[:, :3]
    t = T_cw[:, 3]
    T44 = np.eye(4)
    T44[:3, :3] = R.T
    T44[:3,  3] = -R.T @ t
    return T44


def reprojection_error_graph(db: TrackingDB, min_length: int = 6,
                              out_path: str = 'out_reprojection.png') -> None:
    """
    Pick a track of length >= min_length.
    Triangulate its 3D point using the last frame's stereo pair and GT cameras.
    Project to every frame of the track (left + right) and plot L2 reprojection error.
    """
    track_id = next(
        (t for t in db.all_tracks() if len(db.frames(t)) >= min_length), None
    )
    if track_id is None:
        print(f"No track of length >= {min_length} found.")
        return

    frame_ids = db.frames(track_id)
    gt_poses  = _load_gt_poses(GT_PATH)   # list of 3×4 matrices
    P0, P1    = read_cameras(CALIB_PATH)  # 3×4 calibration projection matrices

    # ── triangulate from the last frame ──────────────────────────────────────
    last_fid = frame_ids[-1]
    lnk_last = db.link(last_fid, track_id)

    T_wc_last = _world_to_cam_44(gt_poses[last_fid])
    P_left_last  = P0 @ T_wc_last   # 3×4
    P_right_last = P1 @ T_wc_last   # 3×4

    pt_left  = np.array([[lnk_last.x_left],  [lnk_last.y]], dtype=np.float64)
    pt_right = np.array([[lnk_last.x_right], [lnk_last.y]], dtype=np.float64)

    pts4d = cv2.triangulatePoints(P_left_last, P_right_last, pt_left, pt_right)
    w = pts4d[3, 0]
    X_world = pts4d[:3, 0] / w                # 3-vector in world coords
    X_world_h = np.append(X_world, 1.0)       # homogeneous

    # ── reproject and measure error for every frame ───────────────────────────
    errors_left, errors_right = [], []

    for fid in frame_ids:
        lnk = db.link(fid, track_id)

        T_wc = _world_to_cam_44(gt_poses[fid])
        P_left_i  = P0 @ T_wc
        P_right_i = P1 @ T_wc

        # left
        proj_l = P_left_i @ X_world_h
        proj_l = proj_l[:2] / proj_l[2]
        errors_left.append(np.linalg.norm(proj_l - [lnk.x_left, lnk.y]))

        # right
        proj_r = P_right_i @ X_world_h
        proj_r = proj_r[:2] / proj_r[2]
        errors_right.append(np.linalg.norm(proj_r - [lnk.x_right, lnk.y]))

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frame_ids, errors_left,  marker='o', label='Left camera',  color='steelblue')
    ax.plot(frame_ids, errors_right, marker='s', label='Right camera', color='tomato')
    ax.axvline(last_fid, color='gray', linestyle='--', linewidth=1,
               label=f'Triangulation frame ({last_fid})')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Reprojection error (pixels)')
    ax.set_title(f'Reprojection error — Track {track_id} (length={len(frame_ids)})')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"Reprojection error graph saved to {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Building TrackingDB over {NUM_FRAMES} KITTI stereo frames …\n")
    db = build_tracking_db(NUM_FRAMES)

    demo_queries(db)
    tracking_statistics(db)
    display_track(db, min_length=6)
    connectivity_graph(db)
    inlier_percentage_graph(db)
    track_length_histogram(db)
    reprojection_error_graph(db, min_length=6)
    demo_serialization(db)

    print("\n─── Consistency check ──────────────────────────────────────")
    db._check_consistency()
