import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
from sift_features import read_images, run_sift

DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'dataset')
CALIB_PATH   = os.path.join(DATASET_PATH, 'sequences', '00', 'calib.txt')
EPIPOLAR_THRESH = 2.0


def read_cameras(calib_path):
    """Return P0 (left) and P1 (right) as 3x4 numpy arrays."""
    with open(calib_path) as f:
        lines = f.readlines()
    matrices = {}
    for line in lines:
        key, *vals = line.split()
        matrices[key.rstrip(':')] = np.array(vals, dtype=float).reshape(3, 4)
    return matrices['P0'], matrices['P1']


def triangulate_stereo_point(P1, P2, point1, point2):
    u1, v1 = point1
    u2, v2 = point2

    p1_row1, p1_row2, p1_row3 = P1[0], P1[1], P1[2]
    p2_row1, p2_row2, p2_row3 = P2[0], P2[1], P2[2]

    A = np.array([
        u1 * p1_row3 - p1_row1,
        v1 * p1_row3 - p1_row2,
        u2 * p2_row3 - p2_row1,
        v2 * p2_row3 - p2_row2,
    ])

    _, _, Vt = np.linalg.svd(A)
    X_homogeneous = Vt[-1]

    X_h, Y_h, Z_h, W = X_homogeneous

    if abs(W) < 1e-5:
        return None  # point at infinity

    X, Y, Z = X_h / W, Y_h / W, Z_h / W

    depth_cam1 = p1_row3[0]*X + p1_row3[1]*Y + p1_row3[2]*Z + p1_row3[3]
    depth_cam2 = p2_row3[0]*X + p2_row3[1]*Y + p2_row3[2]*Z + p2_row3[3]

    if depth_cam1 <= 0 or depth_cam2 <= 0:
        return None  # behind one or both cameras

    return np.array([X, Y, Z])


def get_stereo_inliers(img1, kp1, des1, img2, kp2, des2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    return [
        m for m in matches
        if abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1]) <= EPIPOLAR_THRESH
        and kp1[m.queryIdx].pt[0] > kp2[m.trainIdx].pt[0]
    ]


def show_point_cloud(points_3d, title):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 2], points_3d[:, 1],
               s=2, c=points_3d[:, 2], cmap='plasma')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Z / depth (m)')
    ax.set_zlabel('Y (m)')
    ax.set_title(title)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plot_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    cv2.imshow(title, plot_bgr)


FRAME_INDICES = list(range(10))   # frames to process in batch


def triangulate_frame(P1, P2, frame_idx):
    """Triangulate one stereo frame; returns (custom_pts, cv_pts) or (None, None)."""
    img1, img2 = read_images(frame_idx)
    if img1 is None or img2 is None:
        print(f"  [frame {frame_idx}] could not load images, skipping.")
        return None, None

    kp1, des1 = run_sift(img1)
    kp2, des2 = run_sift(img2)
    inliers = get_stereo_inliers(img1, kp1, des1, img2, kp2, des2)
    if not inliers:
        print(f"  [frame {frame_idx}] no inlier matches, skipping.")
        return None, None

    pts1 = np.array([kp1[m.queryIdx].pt for m in inliers])
    pts2 = np.array([kp2[m.trainIdx].pt for m in inliers])

    custom_pts = []
    valid_mask = []
    for pt1, pt2 in zip(pts1, pts2):
        X = triangulate_stereo_point(P1, P2, pt1, pt2)
        valid_mask.append(X is not None)
        if X is not None:
            custom_pts.append(X)

    valid_mask = np.array(valid_mask)
    custom_pts = np.array(custom_pts)

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4d = pts4d[:, valid_mask]
    w = pts4d[3]
    cv_pts = (pts4d[:3] / w).T

    print(f"  [frame {frame_idx}] inliers={len(inliers)}, "
          f"triangulated={len(custom_pts)}, failed={(~valid_mask).sum()}")
    return custom_pts, cv_pts


def main():
    P1, P2 = read_cameras(CALIB_PATH)
    print("P0 (left camera):\n", P1)
    print("P1 (right camera):\n", P2)
    print(f"\nProcessing {len(FRAME_INDICES)} frames: {FRAME_INDICES}")

    all_custom, all_cv = [], []
    for idx in FRAME_INDICES:
        custom_pts, cv_pts = triangulate_frame(P1, P2, idx)
        if custom_pts is not None and len(custom_pts):
            all_custom.append(custom_pts)
            all_cv.append(cv_pts)

    if not all_custom:
        print("No points triangulated across any frame.")
        return None, None

    all_custom = np.vstack(all_custom)
    all_cv     = np.vstack(all_cv)

    distances = np.linalg.norm(all_custom - all_cv, axis=1)
    print(f"\nTotal points — custom: {len(all_custom)}, OpenCV: {len(all_cv)}")
    print(f"Median distance custom vs OpenCV: {np.median(distances):.6f} m")
    print(f"Max distance: {distances.max():.6f} m")

    show_point_cloud(all_cv,     f'OpenCV triangulated 3D points ({len(FRAME_INDICES)} frames)')
    show_point_cloud(all_custom, f'Custom SVD triangulated 3D points ({len(FRAME_INDICES)} frames)')
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return all_custom, all_cv


if __name__ == '__main__':
    main()
