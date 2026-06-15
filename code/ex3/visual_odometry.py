import sys
import os
import glob
import time
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2'))
from sift_features import read_images, run_sift, match_keypoints
from triangulation import get_stereo_inliers, read_cameras, CALIB_PATH

RANSAC_ITERS     = 100
SUPPORTER_THRESH = 2.0
SAVE_EVERY       = 200

GT_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'dataset','poses', '00.txt')


# ── helpers ─────────────────────────────────────────────────────────────────

def _project(P, X):
    x = P @ np.array([X[0], X[1], X[2], 1.0])
    return x[:2] / x[2]


def _build_quads(img_l0, img_r0, img_l1, img_r1,
                 kp_l0, des_l0, kp_r0, des_r0,
                 kp_l1, des_l1, kp_r1, des_r1,
                 P0, P1):
    good, _, _ = match_keypoints(img_l0, kp_l0, des_l0, img_l1, kp_l1, des_l1, ratio=0.8)
    s0 = get_stereo_inliers(img_l0, kp_l0, des_l0, img_r0, kp_r0, des_r0)
    s1 = get_stereo_inliers(img_l1, kp_l1, des_l1, img_r1, kp_r1, des_r1)
    lr0 = {m.queryIdx: m.trainIdx for m in s0}
    lr1 = {m.queryIdx: m.trainIdx for m in s1}
    quads = []
    for m in good:
        li, ri = m.queryIdx, m.trainIdx
        if li not in lr0 or ri not in lr1:
            continue
        pl = np.array(kp_l0[li].pt, dtype=np.float64).reshape(2, 1)
        pr = np.array(kp_r0[lr0[li]].pt, dtype=np.float64).reshape(2, 1)
        p4 = cv2.triangulatePoints(P0, P1, pl, pr)
        w  = p4[3, 0]
        if abs(w) < 1e-6:
            continue
        X3d = p4[:3, 0] / w
        if X3d[2] > 0:
            quads.append((li, ri, X3d))
    return quads


def ransac_pnp_precomputed(img_l0, img_r0, img_l1, img_r1,
                            kp_l0, des_l0, kp_r0, des_r0,
                            kp_l1, des_l1, kp_r1, des_r1,
                            P0, P1, K, n_iters=RANSAC_ITERS,
                            thresh=SUPPORTER_THRESH, rng=None):
    """RANSAC-PnP with pre-computed SIFT. Returns (R, t, n_inliers)."""
    if rng is None:
        rng = np.random.default_rng()

    quads = _build_quads(img_l0, img_r0, img_l1, img_r1,
                         kp_l0, des_l0, kp_r0, des_r0,
                         kp_l1, des_l1, kp_r1, des_r1,
                         P0, P1)
    n = len(quads)
    if n < 4:
        return None, None, 0

    dist = np.zeros((4, 1))
    best_inliers, best_R, best_t = [], None, None

    for _ in range(n_iters):
        idx  = rng.choice(n, size=4, replace=False)
        s3d  = np.array([quads[j][2] for j in idx], dtype=np.float64)
        s2d  = np.array([kp_l1[quads[j][1]].pt for j in idx], dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(s3d, s2d, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.flatten()
        P_L1 = K @ np.hstack([R, t.reshape(3, 1)])
        inliers = [i for i, (_, l1i, X) in enumerate(quads)
                   if np.linalg.norm(_project(P_L1, X) - np.array(kp_l1[l1i].pt)) < thresh]
        if len(inliers) > len(best_inliers):
            best_inliers, best_R, best_t = inliers, R, t

    if len(best_inliers) < 4:
        return None, None, 0

    # refit on all inliers
    inl3d = np.array([quads[j][2] for j in best_inliers], dtype=np.float64)
    inl2d = np.array([kp_l1[quads[j][1]].pt for j in best_inliers], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(inl3d, inl2d, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if ok:
        best_R, _ = cv2.Rodrigues(rvec)
        best_t    = tvec.flatten()

    return best_R, best_t, len(best_inliers)


# ── ground truth ─────────────────────────────────────────────────────────────

def load_ground_truth(gt_path):
    """
    Parse KITTI poses/00.txt.
    Each line: 12 numbers = 3x4 cam-to-world matrix [R | t] in row-major.
    Camera centre in world coords: C = t  (4th column directly).
    Returns (N, 3) array of camera centres.
    """
    centres = []
    with open(gt_path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            T = np.array(vals).reshape(3, 4)
            centres.append(T[:, 3])
    return np.array(centres)


# ── main odometry loop ────────────────────────────────────────────────────────

def run_odometry(start_frame=0, num_frames=200,
                 traj_path='trajectory.npy', extrinsics_path='extrinsics.npy'):
    P0, P1 = read_cameras(CALIB_PATH)
    K      = P0[:3, :3]
    rng    = np.random.default_rng(42)

    print(f"Sequence 00: frames {start_frame}–{start_frame + num_frames - 1}  "
          f"({num_frames - 1} pairs)")

    img_l0, img_r0 = read_images(start_frame)
    kp_l0, des_l0  = run_sift(img_l0)
    kp_r0, des_r0  = run_sift(img_r0)

    rel_poses = [np.eye(4)]   # T_rel per step; identity for frame 0
    n_failed  = 0
    t0        = time.time()

    for x in range(num_frames - 1):
        img_l1, img_r1 = read_images(start_frame + x + 1)
        kp_l1, des_l1  = run_sift(img_l1)
        kp_r1, des_r1  = run_sift(img_r1)

        R, t, n_inl = ransac_pnp_precomputed(
            img_l0, img_r0, img_l1, img_r1,
            kp_l0, des_l0, kp_r0, des_r0,
            kp_l1, des_l1, kp_r1, des_r1,
            P0, P1, K, rng=rng)

        T_rel = np.eye(4)
        if R is not None:
            T_rel[:3, :3] = R
            T_rel[:3,  3] = t
        else:
            n_failed += 1

        rel_poses.append(T_rel)

        # slide window
        img_l0, img_r0 = img_l1, img_r1
        kp_l0, des_l0  = kp_l1, des_l1
        kp_r0, des_r0  = kp_r1, des_r1

        if (x + 1) % 100 == 0:
            elapsed = time.time() - t0
            fps     = (x + 1) / elapsed
            eta     = (num_frames - 2 - x) / max(fps, 1e-6)
            print(f"  [{x+1:4d}/{num_frames-1}]  inliers={n_inl:3d}  "
                  f"failed={n_failed}  {elapsed:.0f}s elapsed  ETA {eta:.0f}s")

        if (x + 1) % SAVE_EVERY == 0:
            np.save(extrinsics_path, np.array(rel_poses))

    rel = np.array(rel_poses)
    np.save(extrinsics_path, rel)

    # compose relative poses once to build the trajectory
    traj      = [np.zeros(3)]
    T_global  = np.eye(4)
    for T_rel in rel_poses[1:]:
        T_global = T_rel @ T_global
        traj.append(-T_global[:3, :3].T @ T_global[:3, 3])
    traj = np.array(traj)
    np.save(traj_path, traj)

    print(f"\nDone — {num_frames-1} pairs, {n_failed} failed.")
    print(f"Saved: {traj_path}  |  {extrinsics_path}")
    return traj, rel


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory(traj_est, traj_gt, out_path='out_full_trajectory.png'):
    fig, ax = plt.subplots(figsize=(12, 9))

    # estimated
    ax.plot(traj_est[:, 0], traj_est[:, 2],
            linewidth=0.8, color='steelblue', alpha=0.8, label='Estimated')
    ax.scatter(traj_est[0,  0], traj_est[0,  2], s=80, color='blue',  zorder=5)
    ax.scatter(traj_est[-1, 0], traj_est[-1, 2], s=80, color='blue',  zorder=5,
               marker='s')

    # ground truth
    gt = traj_gt[:len(traj_est)]     # align lengths
    ax.plot(gt[:, 0], gt[:, 2],
            linewidth=0.8, color='tomato', alpha=0.8, label='Ground truth')
    ax.scatter(gt[0,  0], gt[0,  2],  s=80, color='red', zorder=5)
    ax.scatter(gt[-1, 0], gt[-1, 2],  s=80, color='red', zorder=5, marker='s')

    ax.set_xlabel('X  (m)')
    ax.set_ylabel('Z / forward  (m)')
    ax.set_title(f'KITTI seq 00 — estimated vs ground-truth trajectory\n'
                 f'{len(traj_est)} frames  (bird\'s-eye, X-Z plane)')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f'Plot saved to {out_path}')


if __name__ == '__main__':
    START_FRAME = 0
    NUM_FRAMES  = 2000

    traj, extr = run_odometry(start_frame=START_FRAME, num_frames=NUM_FRAMES)
    gt_full    = load_ground_truth(GT_PATH)
    gt_window  = gt_full[START_FRAME: START_FRAME + NUM_FRAMES]

    # shift GT so it starts at the origin, same as the estimated trajectory
    gt_shifted = gt_window - gt_window[0]
    plot_trajectory(traj, gt_shifted)
