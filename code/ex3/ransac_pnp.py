import sys
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2'))
from sift_features import read_images, run_sift, match_keypoints
from triangulation import get_stereo_inliers, read_cameras, CALIB_PATH

NUM_FRAMES      = 10    # number of consecutive pairs to process
SUPPORTER_THRESH = 2.0  # pixels


def _build_quads(img_l0, img_r0, img_l1, img_r1,
                 kp_l0, des_l0, kp_r0, des_r0,
                 kp_l1, des_l1, kp_r1, des_r1,
                 P0, P1):
    """
    Return quadruples (l0_idx, l1_idx, r0_idx, r1_idx, X3d) for every
    temporal match whose L0 keypoint also has a stereo match in R0 and
    whose L1 keypoint has a stereo match in R1.
    """
    good_temporal, _, _ = match_keypoints(
        img_l0, kp_l0, des_l0, img_l1, kp_l1, des_l1, ratio=0.8)
    stereo0  = get_stereo_inliers(img_l0, kp_l0, des_l0, img_r0, kp_r0, des_r0)
    stereo1  = get_stereo_inliers(img_l1, kp_l1, des_l1, img_r1, kp_r1, des_r1)
    l0_to_r0 = {m.queryIdx: m.trainIdx for m in stereo0}
    l1_to_r1 = {m.queryIdx: m.trainIdx for m in stereo1}

    quads = []
    for m in good_temporal:
        l0_idx, l1_idx = m.queryIdx, m.trainIdx
        if l0_idx not in l0_to_r0 or l1_idx not in l1_to_r1:
            continue
        r0_idx = l0_to_r0[l0_idx]
        pt_l = np.array(kp_l0[l0_idx].pt, dtype=np.float64).reshape(2, 1)
        pt_r = np.array(kp_r0[r0_idx].pt, dtype=np.float64).reshape(2, 1)
        pt4d = cv2.triangulatePoints(P0, P1, pt_l, pt_r)
        w = pt4d[3, 0]
        if abs(w) < 1e-6:
            continue
        X3d = pt4d[:3, 0] / w
        if X3d[2] <= 0:
            continue
        quads.append((l0_idx, l1_idx, r0_idx, l1_to_r1[l1_idx], X3d))
    return quads


def get_pose_pnp(frame_x, P0, P1, K):
    """
    Estimate the pose of frame X+1 relative to frame X using PnP (no RANSAC).
    Uses quadruple correspondences (seen in all 4 images) for robustness.

    Returns R (3x3), t (3,) and the number of correspondences used.
    """
    frame_x1 = frame_x + 1
    img_l0, img_r0 = read_images(frame_x)
    img_l1, img_r1 = read_images(frame_x1)
    if img_l0 is None or img_r0 is None or img_l1 is None or img_r1 is None:
        return None, None, 0

    kp_l0, des_l0 = run_sift(img_l0)
    kp_r0, des_r0 = run_sift(img_r0)
    kp_l1, des_l1 = run_sift(img_l1)
    kp_r1, des_r1 = run_sift(img_r1)

    quads = _build_quads(img_l0, img_r0, img_l1, img_r1,
                         kp_l0, des_l0, kp_r0, des_r0,
                         kp_l1, des_l1, kp_r1, des_r1,
                         P0, P1)
    n = len(quads)
    if n < 4:
        print(f"  [pair {frame_x}-{frame_x1}] not enough correspondences ({n}), skipping.")
        return None, None, n

    pts_3d = np.array([q[4] for q in quads], dtype=np.float64)
    pts_2d = np.array([kp_l1[q[1]].pt for q in quads], dtype=np.float64)
    dist   = np.zeros((4, 1))

    ok, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None, n

    R, _ = cv2.Rodrigues(rvec)
    t    = tvec.flatten()
    return R, t, n


def _project(P, X):
    x = P @ np.array([X[0], X[1], X[2], 1.0])
    return x[:2] / x[2]


def find_supporters(frame_x, P0, P1, K, R, t, thresh=SUPPORTER_THRESH):
    """
    For the transformation T=[R|t] (pose of frame X+1 in frame X coords),
    classify every quadruple match as supporter or non-supporter based on
    whether the triangulated 3D point reprojects within `thresh` pixels in
    all four images.

    Returns (supporters, non_supporters, kp_l0, kp_l1, img_l0, img_l1)
    where supporters / non_supporters are lists of (l0_idx, l1_idx).
    """
    frame_x1 = frame_x + 1
    img_l0, img_r0 = read_images(frame_x)
    img_l1, img_r1 = read_images(frame_x1)

    kp_l0, des_l0 = run_sift(img_l0)
    kp_r0, des_r0 = run_sift(img_r0)
    kp_l1, des_l1 = run_sift(img_l1)
    kp_r1, des_r1 = run_sift(img_r1)

    quads = _build_quads(img_l0, img_r0, img_l1, img_r1,
                         kp_l0, des_l0, kp_r0, des_r0,
                         kp_l1, des_l1, kp_r1, des_r1,
                         P0, P1)

    t_stereo = np.array([P1[0, 3] / K[0, 0], 0., 0.])
    P_L0 = P0
    P_R0 = P1
    P_L1 = K @ np.hstack([R, t.reshape(3, 1)])
    P_R1 = K @ np.hstack([R, (t + t_stereo).reshape(3, 1)])

    supporters, non_supporters = [], []
    for l0_idx, l1_idx, r0_idx, r1_idx, X3d in quads:
        pt_l0 = np.array(kp_l0[l0_idx].pt)
        pt_r0 = np.array(kp_r0[r0_idx].pt)
        pt_l1 = np.array(kp_l1[l1_idx].pt)
        pt_r1 = np.array(kp_r1[r1_idx].pt)

        d_l0 = np.linalg.norm(_project(P_L0, X3d) - pt_l0)
        d_r0 = np.linalg.norm(_project(P_R0, X3d) - pt_r0)
        d_l1 = np.linalg.norm(_project(P_L1, X3d) - pt_l1)
        d_r1 = np.linalg.norm(_project(P_R1, X3d) - pt_r1)

        if d_l0 < thresh and d_r0 < thresh and d_l1 < thresh and d_r1 < thresh:
            supporters.append((l0_idx, l1_idx))
        else:
            non_supporters.append((l0_idx, l1_idx))

    return supporters, non_supporters, kp_l0, kp_l1, img_l0, img_l1


def _count_supporters(quads, kp_l0, kp_r0, kp_l1, kp_r1,
                      P_L0, P_R0, P_L1, P_R1, thresh):
    """Return list of quad indices whose 3D point reprojects within thresh in all 4 images."""
    inliers = []
    for i, (l0_idx, l1_idx, r0_idx, r1_idx, X3d) in enumerate(quads):
        d_l0 = np.linalg.norm(_project(P_L0, X3d) - np.array(kp_l0[l0_idx].pt))
        d_r0 = np.linalg.norm(_project(P_R0, X3d) - np.array(kp_r0[r0_idx].pt))
        d_l1 = np.linalg.norm(_project(P_L1, X3d) - np.array(kp_l1[l1_idx].pt))
        d_r1 = np.linalg.norm(_project(P_R1, X3d) - np.array(kp_r1[r1_idx].pt))
        if d_l0 < thresh and d_r0 < thresh and d_l1 < thresh and d_r1 < thresh:
            inliers.append(i)
    return inliers


def _build_proj_matrices(R, t, P0, P1, K):
    t_stereo = np.array([P1[0, 3] / K[0, 0], 0., 0.])
    P_L1 = K @ np.hstack([R, t.reshape(3, 1)])
    P_R1 = K @ np.hstack([R, (t + t_stereo).reshape(3, 1)])
    return P0, P1, P_L1, P_R1


def ransac_pnp(frame_x, P0, P1, K,
               n_iters=1000, thresh=SUPPORTER_THRESH, seed=42):
    """
    RANSAC-PnP for the pair (frame_x, frame_x+1).

    Each iteration:
      1. Sample 4 random quadruple correspondences.
      2. Solve PnP (SQPNP) on those 4 points.
      3. Count supporters across all 4 images.

    After all iterations, refit PnP on ALL inliers of the best model.

    Returns
    -------
    R, t        : refined pose (3x3, 3,)
    best_inlier_indices : indices into quads for the best model's supporters
    n_supporters : number of supporters of the refined model
    quads        : full quadruple list (so callers can retrieve kp info)
    kp_l0, kp_r0, kp_l1, kp_r1, img_l0, img_l1
    """
    rng = np.random.default_rng(seed)
    frame_x1 = frame_x + 1

    img_l0, img_r0 = read_images(frame_x)
    img_l1, img_r1 = read_images(frame_x1)
    kp_l0, des_l0 = run_sift(img_l0)
    kp_r0, des_r0 = run_sift(img_r0)
    kp_l1, des_l1 = run_sift(img_l1)
    kp_r1, des_r1 = run_sift(img_r1)

    quads = _build_quads(img_l0, img_r0, img_l1, img_r1,
                         kp_l0, des_l0, kp_r0, des_r0,
                         kp_l1, des_l1, kp_r1, des_r1,
                         P0, P1)
    n = len(quads)
    if n < 4:
        return None, None, [], 0, quads, kp_l0, kp_r0, kp_l1, kp_r1, img_l0, img_l1

    dist = np.zeros((4, 1))
    best_inliers = []
    best_R, best_t = None, None

    for i in range(n_iters):
        # --- sample 4 correspondences ---
        sample_idx = rng.choice(n, size=4, replace=False)
        s_3d = np.array([quads[j][4] for j in sample_idx], dtype=np.float64)
        s_2d = np.array([kp_l1[quads[j][1]].pt for j in sample_idx], dtype=np.float64)

        ok, rvec, tvec = cv2.solvePnP(s_3d, s_2d, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.flatten()

        P_L0, P_R0, P_L1, P_R1 = _build_proj_matrices(R, t, P0, P1, K)
        inliers = _count_supporters(quads, kp_l0, kp_r0, kp_l1, kp_r1,
                                    P_L0, P_R0, P_L1, P_R1, thresh)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_R, best_t = R, t

        if (i + 1) % 100 == 0:
            print(f"  iter {i+1:4d}/{n_iters}  best supporters so far: {len(best_inliers)}")

    # --- refit on ALL inliers of the best model ---
    print(f"  Best model: {len(best_inliers)} supporters out of {n} quads")
    if len(best_inliers) >= 4:
        inl_3d = np.array([quads[j][4] for j in best_inliers], dtype=np.float64)
        inl_2d = np.array([kp_l1[quads[j][1]].pt for j in best_inliers], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(inl_3d, inl_2d, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if ok:
            best_R, _ = cv2.Rodrigues(rvec)
            best_t    = tvec.flatten()
            # recount supporters with the refined model
            P_L0, P_R0, P_L1, P_R1 = _build_proj_matrices(best_R, best_t, P0, P1, K)
            best_inliers = _count_supporters(quads, kp_l0, kp_r0, kp_l1, kp_r1,
                                             P_L0, P_R0, P_L1, P_R1, thresh)
            print(f"  After refit on inliers: {len(best_inliers)} supporters")

    return (best_R, best_t, best_inliers, len(best_inliers),
            quads, kp_l0, kp_r0, kp_l1, kp_r1, img_l0, img_l1)


def plot_supporters(supporters, non_supporters, kp_l0, kp_l1, img_l0, img_l1,
                    out_path='out_supporters.png'):
    H, W = img_l0.shape
    canvas = np.hstack([cv2.cvtColor(img_l0, cv2.COLOR_GRAY2BGR),
                        cv2.cvtColor(img_l1, cv2.COLOR_GRAY2BGR)])

    COL_SUP = (0, 200,   0)   # green
    COL_NON = (0,   0, 220)   # red

    for l0_idx, l1_idx in non_supporters:
        p0 = tuple(map(int, kp_l0[l0_idx].pt))
        p1 = (int(kp_l1[l1_idx].pt[0]) + W, int(kp_l1[l1_idx].pt[1]))
        cv2.line(canvas, p0, p1, COL_NON, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, COL_NON, -1)
        cv2.circle(canvas, p1, 3, COL_NON, -1)

    for l0_idx, l1_idx in supporters:
        p0 = tuple(map(int, kp_l0[l0_idx].pt))
        p1 = (int(kp_l1[l1_idx].pt[0]) + W, int(kp_l1[l1_idx].pt[1]))
        cv2.line(canvas, p0, p1, COL_SUP, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 4, COL_SUP, -1)
        cv2.circle(canvas, p1, 4, COL_SUP, -1)

    cv2.line(canvas, (W, 0), (W, H), (200, 200, 200), 1)
    cv2.putText(canvas, 'Left frame 0', (10,    28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(canvas, 'Left frame 1', (W + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
    cv2.putText(canvas, f'Supporters (green): {len(supporters)}',
                (10, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_SUP, 2)
    cv2.putText(canvas, f'Non-supporters (red): {len(non_supporters)}',
                (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_NON, 2)

    cv2.imwrite(out_path, canvas)
    print(f'Saved {out_path}')


def _triangulate_stereo_cloud(img_l, img_r, kp_l, des_l, kp_r, des_r, P0, P1):
    """Triangulate all stereo inlier matches for one frame. Returns (N,3) array."""
    inliers = get_stereo_inliers(img_l, kp_l, des_l, img_r, kp_r, des_r)
    pts = []
    for m in inliers:
        pt_l = np.array(kp_l[m.queryIdx].pt, dtype=np.float64).reshape(2, 1)
        pt_r = np.array(kp_r[m.trainIdx].pt, dtype=np.float64).reshape(2, 1)
        pt4d = cv2.triangulatePoints(P0, P1, pt_l, pt_r)
        w = pt4d[3, 0]
        if abs(w) < 1e-6:
            continue
        X = pt4d[:3, 0] / w
        if X[2] > 0:
            pts.append(X)
    return np.array(pts)


def plot_point_clouds(frame_x, P0, P1, K, R, t, out_path='out_point_clouds.png'):
    """
    Plot two 3D point clouds from above (X-Z plane):
      - Cloud 0: stereo triangulation of frame X           (blue)
      - Cloud 1: stereo triangulation of frame X+1,        (orange)
                 transformed into frame X coords via T^{-1}
    Points far away or behind the camera are cropped out.
    """
    frame_x1 = frame_x + 1
    img_l0, img_r0 = read_images(frame_x)
    img_l1, img_r1 = read_images(frame_x1)

    kp_l0, des_l0 = run_sift(img_l0)
    kp_r0, des_r0 = run_sift(img_r0)
    kp_l1, des_l1 = run_sift(img_l1)
    kp_r1, des_r1 = run_sift(img_r1)

    # Cloud 0 after T: apply T to bring frame 0 points into frame 1's camera coords
    # T: X_cam1 = R @ X_world + t
    cloud0_world = _triangulate_stereo_cloud(img_l0, img_r0, kp_l0, des_l0, kp_r0, des_r0, P0, P1)
    cloud0_after_T = (R @ cloud0_world.T + t.reshape(3, 1)).T

    # Cloud 1: triangulated directly in frame 1's camera coords
    cloud1 = _triangulate_stereo_cloud(img_l1, img_r1, kp_l1, des_l1, kp_r1, des_r1, P0, P1)

    def crop(pts, z_min=1.0, z_max=60.0, x_abs_max=40.0, y_abs_max=10.0):
        mask = ((pts[:, 2] > z_min) & (pts[:, 2] < z_max) &
                (np.abs(pts[:, 0]) < x_abs_max) &
                (np.abs(pts[:, 1]) < y_abs_max))
        return pts[mask]

    cloud0_after_T = crop(cloud0_after_T)
    cloud1         = crop(cloud1)
    print(f"  Cloud 0 after T:  {len(cloud0_after_T)} pts after crop")
    print(f"  Cloud 1 (pair {frame_x1}): {len(cloud1)} pts after crop")

    fig = plt.figure(figsize=(12, 7))
    ax  = fig.add_subplot(111, projection='3d')

    ax.scatter(cloud0_after_T[:, 0], cloud0_after_T[:, 2], -cloud0_after_T[:, 1],
               s=1, color='steelblue', alpha=0.4,
               label=f'Cloud 0 after T (frame {frame_x})')
    ax.scatter(cloud1[:, 0], cloud1[:, 2], -cloud1[:, 1],
               s=1, color='darkorange', alpha=0.4,
               label=f'Cloud 1 (frame {frame_x1})')

    # camera positions (L1 at origin, L0 behind it)
    cam0 = t
    ax.scatter([0],        [0],           [0],
               s=60, color='orange', marker='^', zorder=5, label=f'Cam L{frame_x1}')
    ax.scatter([cam0[0]], [cam0[2]], [-cam0[1]],
               s=60, color='blue',   marker='^', zorder=5, label=f'Cam L{frame_x}')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / forward (m)')
    ax.set_zlabel('Y / up (m)')
    ax.set_title(f'Cloud 0 after T (blue) vs Cloud 1 (orange) — frame {frame_x1} coords\n3D view')
    ax.legend(markerscale=6, loc='upper left')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f'  Saved {out_path}')


def main():
    P0, P1 = read_cameras(CALIB_PATH)
    K       = P0[:3, :3]

    print(f"Processing {NUM_FRAMES} consecutive frame pairs\n")
    print(f"{'Pair':>8}  {'pts':>5}  {'tx':>9}  {'ty':>9}  {'tz':>9}  R (row0)  R (row1)  R (row2)")
    print("-" * 95)

    results = []
    for x in range(NUM_FRAMES):
        R, t, n = get_pose_pnp(x, P0, P1, K)
        results.append((x, R, t, n))
        if R is None:
            print(f"  {x:03d}-{x+1:03d}   FAILED  (n={n})")
            continue
        print(f"  {x:03d}-{x+1:03d}  {n:5d}"
              f"  {t[0]:+9.5f}  {t[1]:+9.5f}  {t[2]:+9.5f}"
              f"  [{R[0,0]:+.4f} {R[0,1]:+.4f} {R[0,2]:+.4f}]"
              f"  [{R[1,0]:+.4f} {R[1,1]:+.4f} {R[1,2]:+.4f}]"
              f"  [{R[2,0]:+.4f} {R[2,1]:+.4f} {R[2,2]:+.4f}]")

    # Supporters for frame pair 0->1 (no RANSAC)
    R0, t0, _ = results[0][1], results[0][2], results[0][3]
    if R0 is not None:
        print(f"\nFinding supporters for pair 0->1 (thresh={SUPPORTER_THRESH}px) ...")
        sup, non_sup, kp_l0, kp_l1, img_l0, img_l1 = find_supporters(0, P0, P1, K, R0, t0)
        print(f"  Supporters: {len(sup)},  Non-supporters: {len(non_sup)}")
        plot_supporters(sup, non_sup, kp_l0, kp_l1, img_l0, img_l1,
                        out_path='out_supporters_no_ransac.png')

    # RANSAC-PnP for pair 0->1
    print(f"\nRANSAC-PnP for pair 0->1 (n_iters=1000, thresh={SUPPORTER_THRESH}px) ...")
    (R_rans, t_rans, best_inliers, n_sup,
     quads, kp_l0, kp_r0, kp_l1, kp_r1, img_l0, img_l1) = ransac_pnp(0, P0, P1, K, n_iters=1000)

    if R_rans is not None:
        print(f"  RANSAC t = [{t_rans[0]:+.5f}, {t_rans[1]:+.5f}, {t_rans[2]:+.5f}]")
        inlier_set = set(best_inliers)
        supporters     = [(quads[i][0], quads[i][1]) for i in inlier_set]
        non_supporters = [(quads[i][0], quads[i][1]) for i in range(len(quads))
                          if i not in inlier_set]
        plot_supporters(supporters, non_supporters, kp_l0, kp_l1, img_l0, img_l1,
                        out_path='out_supporters_ransac.png')

        # Point cloud comparison
        print("\nPlotting 3D point clouds ...")
        plot_point_clouds(0, P0, P1, K, R_rans, t_rans)

    return results


if __name__ == '__main__':
    main()
