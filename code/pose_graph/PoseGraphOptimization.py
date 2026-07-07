import sys
import os
import gc
import pickle
import numpy as np
import gtsam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex2'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex4'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex5'))

from triangulation import read_cameras, CALIB_PATH
from tracking_database import TrackingDB
from BundleAdjustment import (
    build_global_poses, make_stereo_cal, build_bundle_window,
    optimize_bundle_window,
    EXTRINSICS_PATH, TRACKING_DB_PATH, BUNDLE_SIZE, BA_ITERATIONS,
)

EX6_DIR   = os.path.dirname(__file__)
CACHE_PATH = os.path.join(EX6_DIR, 'ba_cache.pkl')


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _save_cache(first_bundle, all_relative):
    """
    Persist the computed outputs to disk so the full BA never needs to re-run.

    Saved data (pure numpy — no GTSAM objects):
      first_bundle: {
          'window_fids': list[int],
          'cam_positions': {fid: ndarray(3)},   # camera centres
          'cov_map':       {fid: ndarray(6,6)}, # marginal covariances
          'T_rel':         ndarray(4,4),         # relative pose matrix
          'Sigma_rel':     ndarray(6,6),
      }
      all_relative: [(fc0, fck, ndarray(4,4), ndarray(6,6)), ...]
    """
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump({'first_bundle': first_bundle,
                     'all_relative': all_relative}, f)
    print(f"Cache saved → {CACHE_PATH}")


def _load_cache():
    """Return (first_bundle, all_relative) from cache, or (None, None)."""
    if not os.path.exists(CACHE_PATH):
        return None, None
    with open(CACHE_PATH, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded cached results from {CACHE_PATH}")
    return data['first_bundle'], data['all_relative']


# ── Marginal-covariance helpers ───────────────────────────────────────────────

def get_marginal_covariances(graph, result):
    """
    Build gtsam.Marginals and return (marginals, {fid: cov_6x6}) for every
    pose variable ('x') in result.

    GTSAM Pose3 tangent-space layout:
      indices 0:3  →  rotation  (so(3))
      indices 3:6  →  translation
    """
    marginals = gtsam.Marginals(graph, result)
    cov_map = {}
    for key in result.keys():
        if gtsam.symbolChr(key) == ord('x'):
            fid = gtsam.symbolIndex(key)
            try:
                cov_map[fid] = np.array(marginals.marginalCovariance(key))
            except Exception:
                pass
    return marginals, cov_map


def _joint_cross_cov(marginals, key0, keyk):
    """
    Return (Sigma_00, Sigma_kk, Sigma_0k) extracted from the joint marginal
    of [key0, keyk].  Sigma_0k is the 6×6 cross-covariance block.
    """
    Sigma_00 = np.array(marginals.marginalCovariance(key0))
    Sigma_kk = np.array(marginals.marginalCovariance(keyk))

    kv = gtsam.KeyVector()
    kv.append(key0)
    kv.append(keyk)
    joint = marginals.jointMarginalCovariance(kv)

    # JointMarginal.at(row_key, col_key) returns the block at that position
    Sigma_0k = np.array(joint.at(key0, keyk))
    return Sigma_00, Sigma_kk, Sigma_0k


def compute_relative_pose_and_cov(marginals, result, fid0, fidk):
    """
    Relative pose T(ck|c0) and conditional covariance Σ(ck|c0).

    Algorithm:
      1. Marginalise the posterior to the joint 12×12 covariance of [c0, ck].
      2. Condition on c0  →  Schur complement:
             Σ_rel = Σ_kk − Σ_k0 · Σ_00^{-1} · Σ_0k

    The result is the covariance associated with P(ck | c0), i.e. the
    uncertainty of the relative pose once c0 is treated as perfectly known.

    Returns (T_rel: Pose3, Sigma_rel: ndarray 6×6), or (None, None).
    """
    key0 = gtsam.symbol('x', fid0)
    keyk = gtsam.symbol('x', fidk)

    if not result.exists(key0) or not result.exists(keyk):
        return None, None

    pose_c0 = result.atPose3(key0)
    pose_ck = result.atPose3(keyk)
    T_rel   = pose_c0.between(pose_ck)      # pose of ck expressed in c0's frame

    # Marginalisation → joint blocks
    Sigma_00, Sigma_kk, Sigma_0k = _joint_cross_cov(marginals, key0, keyk)
    Sigma_k0 = Sigma_0k.T

    # Conditioning: P(ck | c0)
    Sigma_rel = Sigma_kk - Sigma_k0 @ np.linalg.solve(Sigma_00, Sigma_0k)

    return T_rel, Sigma_rel


# ── 3-D covariance ellipsoid ──────────────────────────────────────────────────

def _draw_ellipsoid_3d(ax, center, cov3x3, n_std=3.0,
                        color='lightblue', alpha=0.25):
    """Render a covariance ellipsoid at n_std standard deviations."""
    try:
        eigvals, eigvecs = np.linalg.eigh(cov3x3)
        eigvals = np.maximum(eigvals, 0.0)
        radii   = n_std * np.sqrt(eigvals)

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 14)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        for i in range(xs.shape[0]):
            for j in range(xs.shape[1]):
                pt = eigvecs @ (radii * np.array([xs[i, j], ys[i, j], zs[i, j]])) + center
                xs[i, j], ys[i, j], zs[i, j] = pt

        ax.plot_surface(xs, ys, zs, color=color, alpha=alpha, linewidth=0)
    except Exception:
        pass


def plot_bundle_3d_with_covariances(window_fids, result, cov_map,
                                     out_path, ellipsoid_scale=100.0):
    """
    3-D scatter of all camera centres in the bundle, with translational
    covariance ellipsoids drawn at 3σ (scaled ×ellipsoid_scale so they are
    visible relative to the camera trajectory).

    The 6×6 marginal covariance follows the GTSAM Pose3 convention:
      rows/cols 0:3 → rotation, rows/cols 3:6 → translation.
    """
    cam_items = [(fid, np.array(result.atPose3(gtsam.symbol('x', fid)).translation()))
                 for fid in sorted(window_fids)
                 if result.exists(gtsam.symbol('x', fid))]

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    if cam_items:
        pts = np.array([t for _, t in cam_items])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=1.5, alpha=0.6)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='steelblue', s=60, zorder=5,
                   label='Camera centres')

        for fid, t in cam_items:
            ax.text(t[0], t[1], t[2], f' {fid}', fontsize=7, color='navy')
            if fid in cov_map:
                cov_t = cov_map[fid][3:6, 3:6]          # translation sub-block
                scaled = cov_t * (ellipsoid_scale ** 2)  # inflate for visibility
                _draw_ellipsoid_3d(ax, t, scaled, n_std=3.0,
                                    color='lightblue', alpha=0.30)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(
        f'First bundle — camera locations with 3σ covariance ellipsoids\n'
        f'frames {window_fids[0]}–{window_fids[-1]}  '
        f'(ellipsoids ×{int(ellipsoid_scale)} for visibility)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"3D+covariance plot saved to {out_path}")


def plot_bundle_3d_with_covariances_np(window_fids, cam_positions, cov_map,
                                        out_path, ellipsoid_scale=100.0):
    """Same as plot_bundle_3d_with_covariances but takes plain numpy dicts."""
    cam_items = [(fid, cam_positions[fid])
                 for fid in sorted(window_fids) if fid in cam_positions]

    fig = plt.figure(figsize=(13, 9))
    ax  = fig.add_subplot(111, projection='3d')

    if cam_items:
        pts = np.array([t for _, t in cam_items])
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'b-', linewidth=1.5, alpha=0.6)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='steelblue', s=60, zorder=5,
                   label='Camera centres')
        for fid, t in cam_items:
            ax.text(t[0], t[1], t[2], f' {fid}', fontsize=7, color='navy')
            if fid in cov_map:
                cov_t  = cov_map[fid][3:6, 3:6]
                scaled = cov_t * (ellipsoid_scale ** 2)
                _draw_ellipsoid_3d(ax, t, scaled, n_std=3.0,
                                    color='lightblue', alpha=0.30)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title(
        f'First bundle — camera locations with 3σ covariance ellipsoids\n'
        f'frames {window_fids[0]}–{window_fids[-1]}  '
        f'(ellipsoids ×{int(ellipsoid_scale)} for visibility)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"3D+covariance plot saved to {out_path}")


# ── Print / plot from cached numpy data ──────────────────────────────────────

def _print_and_plot_from_cache(first_bundle, all_relative):
    """Reproduce all console output and plots from cached numpy results."""
    if not first_bundle:
        print("No first-bundle data available.")
        return

    window_fids   = first_bundle['window_fids']
    cam_positions = first_bundle['cam_positions']
    cov_map       = first_bundle['cov_map']
    T_rel_mat     = first_bundle['T_rel']
    Sigma_rel     = first_bundle['Sigma_rel']
    fid_c0        = window_fids[0]
    fid_ck        = window_fids[-1]

    print("=" * 65)
    print(f"First bundle: c0 = frame {fid_c0},  c_k = frame {fid_ck}")
    print("=" * 65)

    print("\nMarginal translation std  (1-σ, metres)  for all frames:")
    for fid in sorted(cov_map):
        std_t = np.sqrt(np.diag(cov_map[fid][3:6, 3:6]))
        print(f"  frame {fid:4d}:  "
              f"σ_t = [{std_t[0]:.5f}, {std_t[1]:.5f}, {std_t[2]:.5f}]")

    # 3-D plot from cached positions + covariances
    plot_bundle_3d_with_covariances_np(
        window_fids, cam_positions, cov_map,
        out_path=os.path.join(EX6_DIR, 'out_bundle0_3d_with_cov.png'))

    if T_rel_mat is not None and Sigma_rel is not None:
        T_rel  = gtsam.Pose3(T_rel_mat)
        trans  = np.array(T_rel.translation())
        rpy    = T_rel.rotation().rpy()

        print(f"\nRelative pose  T(c{fid_ck} | c{fid_c0}):")
        print(f"  Translation (m) : [{trans[0]:+.4f}, {trans[1]:+.4f}, {trans[2]:+.4f}]")
        print(f"  RPY      (rad)  : [{rpy[0]:+.6f}, {rpy[1]:+.6f}, {rpy[2]:+.6f}]")

        np.set_printoptions(precision=6, suppress=True, linewidth=110)
        print(f"\nConditional covariance  Σ(c{fid_ck} | c{fid_c0})  [6×6]")
        print("  Row/col order: [ωx, ωy, ωz,  tx, ty, tz]")
        print(Sigma_rel)

        std = np.sqrt(np.maximum(np.diag(Sigma_rel), 0.0))
        print(f"\n  1-σ rotation  std (rad): [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
        print(f"  1-σ translation std (m): [{std[3]:.6f}, {std[4]:.6f}, {std[5]:.6f}]")

    print("\n" + "=" * 65)
    print("Relative poses between consecutive keyframes — all bundles")
    print("=" * 65)
    for fc0, fck, T_mat, Sigma_r in all_relative:
        T_r = gtsam.Pose3(T_mat)
        t   = np.array(T_r.translation())
        sd  = np.sqrt(np.maximum(np.diag(Sigma_r), 0.0))
        print(f"  [{fc0:4d} → {fck:4d}]  "
              f"T=({t[0]:+7.3f}, {t[1]:+7.3f}, {t[2]:+7.3f}) m  |  "
              f"σ_t=({sd[3]:.4f}, {sd[4]:.4f}, {sd[5]:.4f}) m  "
              f"σ_r=({sd[0]:.5f}, {sd[1]:.5f}, {sd[2]:.5f}) rad")

    print(f"\nTotal consecutive keyframe constraints: {len(all_relative)}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Try cache first ───────────────────────────────────────────────────────
    first_bundle, all_relative = _load_cache()
    if first_bundle is not None:
        _print_and_plot_from_cache(first_bundle, all_relative)
        return all_relative

    # ── No cache — run BA for all bundles ─────────────────────────────────────
    P0, P1 = read_cameras(CALIB_PATH)
    db      = TrackingDB()
    db.load(TRACKING_DB_PATH)
    global_poses = build_global_poses(EXTRINSICS_PATH)
    n_poses      = len(global_poses)
    cal          = make_stereo_cal(P0, P1)

    # Process each bundle inline — never accumulate all graphs/results in memory.
    prev_result  = None
    start        = 0
    w_idx        = 0
    all_relative = []   # [(fc0, fck, T_4x4, Sigma_6x6), ...]
    n_skipped    = 0
    first_bundle = None   # filled on bundle 0

    print("Running BA + extracting relative poses for all bundles …\n")

    while start < n_poses:
        end         = min(start + BUNDLE_SIZE, n_poses)
        window_fids = list(range(start, end))

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

        print(f"  [{w_idx:3d}] frames {window_fids[0]:4d}–{window_fids[-1]:4d}  "
              f"err {err_before:.2f} → {err_after:.2f}  ({n_lm} landmarks)")

        # ── First bundle: full analysis ───────────────────────────────────────
        if first_bundle is None:
            fid_c0 = window_fids[0]
            fid_ck = window_fids[-1]
            try:
                marginals0, cov_map0 = get_marginal_covariances(graph, result)
                T_rel0, Sigma_rel0   = compute_relative_pose_and_cov(
                    marginals0, result, fid_c0, fid_ck)

                # Convert GTSAM objects → numpy for caching
                cam_positions = {}
                for fid in sorted(window_fids):
                    key = gtsam.symbol('x', fid)
                    if result.exists(key):
                        cam_positions[fid] = np.array(
                            result.atPose3(key).translation())

                first_bundle = {
                    'window_fids':   window_fids,
                    'cam_positions': cam_positions,
                    'cov_map':       cov_map0,
                    'T_rel':         T_rel0.matrix() if T_rel0 else None,
                    'Sigma_rel':     Sigma_rel0,
                }
            except RuntimeError as e:
                print(f"  WARNING: could not compute marginals for first bundle: {e}")
                first_bundle = {}

        # ── Every bundle: relative pose + covariance ──────────────────────────
        fc0 = window_fids[0]
        fck = window_fids[-1]
        try:
            marginals = gtsam.Marginals(graph, result)
            T_r, Sigma_r = compute_relative_pose_and_cov(marginals, result, fc0, fck)
            if T_r is not None:
                all_relative.append((fc0, fck, T_r.matrix(), Sigma_r))
        except RuntimeError:
            n_skipped += 1

        prev_result = result
        del graph, ba_cams, init_vals, result
        gc.collect()
        w_idx += 1

        start = end - 1
        if end == n_poses:
            break

    print(f"\nDone — {w_idx} bundles processed.")
    print(f"Constraints extracted: {len(all_relative)}"
          f"  ({n_skipped} skipped — ill-conditioned)\n")

    # ── Save cache ────────────────────────────────────────────────────────────
    _save_cache(first_bundle, all_relative)

    # ── Print + plot results ──────────────────────────────────────────────────
    _print_and_plot_from_cache(first_bundle, all_relative)
    return all_relative


if __name__ == '__main__':
    main()
