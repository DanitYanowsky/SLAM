import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
from sift_features import read_images, run_sift

EPIPOLAR_THRESH = 2.0


def main():
    img1, img2 = read_images(18)
    kp1, des1 = run_sift(img1)
    kp2, des2 = run_sift(img2)

    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    all_matches = [m for m, _ in flann.knnMatch(des1, des2, k=2)]

    inliers = []
    outliers = []
    for m in all_matches:
        y_diff = abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])
        if y_diff <= EPIPOLAR_THRESH:
            inliers.append(m)
        else:
            outliers.append(m)

    print(f"Total matches:    {len(all_matches)}")
    print(f"Inliers:          {len(inliers)}")
    print(f"Outliers:         {len(outliers)}")
    print(f"Discarded:        {len(outliers) / len(all_matches) * 100:.2f}%")

    # --- histogram ---
    y_diffs = np.array([
        abs(kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1])
        for m in all_matches
    ])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_diffs, bins=50, edgecolor='black')
    ax.axvline(x=EPIPOLAR_THRESH, color='red', linestyle='--', label=f'{EPIPOLAR_THRESH}-pixel threshold')
    ax.set_xlabel('Y-coordinate difference (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of Y-coordinate differences between SIFT matches (frame 0)')
    ax.legend()
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    hist_bgr = cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)
    plt.close(fig)
    cv2.imshow('Y-diff Histogram', hist_bgr)

    # --- dot visualisation on image pair ---
    img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    ORANGE = (0, 165, 255)
    CYAN   = (255, 255, 0)

    for m in outliers:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        cv2.circle(img1_bgr, pt1, 3, CYAN, -1)
        cv2.circle(img2_bgr, pt2, 3, CYAN, -1)

    for m in inliers:
        pt1 = tuple(map(int, kp1[m.queryIdx].pt))
        pt2 = tuple(map(int, kp2[m.trainIdx].pt))
        cv2.circle(img1_bgr, pt1, 3, ORANGE, -1)
        cv2.circle(img2_bgr, pt2, 3, ORANGE, -1)

    combined = np.hstack([img1_bgr, img2_bgr])
    cv2.putText(combined, f'Orange: inliers ({len(inliers)})  Cyan: outliers ({len(outliers)})',
                (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow('Inliers (orange) vs Outliers (cyan)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.close('all')


if __name__ == '__main__':
    main()
