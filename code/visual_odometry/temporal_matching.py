import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'ex1'))
from sift_features import read_images, run_sift, match_keypoints

FRAME_IDX = 0   # X — change to test other pairs


def match_consecutive_left(frame_x, frame_x1, ratio=0.8):
    """Match SIFT features between left camera of frame X and frame X+1."""
    img_x, _ = read_images(frame_x)
    img_x1, _ = read_images(frame_x1)
    if img_x is None or img_x1 is None:
        raise FileNotFoundError(f"Could not load frames {frame_x} or {frame_x1}")

    kp_x,  des_x  = run_sift(img_x)
    kp_x1, des_x1 = run_sift(img_x1)

    print(f"Frame {frame_x:06d}: {len(kp_x)} keypoints")
    print(f"Frame {frame_x1:06d}: {len(kp_x1)} keypoints")

    good, discarded, img_matches = match_keypoints(
        img_x, kp_x, des_x,
        img_x1, kp_x1, des_x1,
        ratio=ratio,
    )

    print(f"Good matches: {len(good)},  Discarded: {len(discarded)}")
    return img_x, kp_x, img_x1, kp_x1, good, discarded, img_matches


def main():
    frame_x  = FRAME_IDX
    frame_x1 = FRAME_IDX + 1

    img_x, kp_x, img_x1, kp_x1, good, discarded, img_matches = match_consecutive_left(
        frame_x, frame_x1
    )

    cv2.imshow(f'Left-camera matches: frame {frame_x} → frame {frame_x1}', img_matches)

    # Side-by-side view with lines for the first N good matches
    img_x_bgr  = cv2.cvtColor(img_x,  cv2.COLOR_GRAY2BGR)
    img_x1_bgr = cv2.cvtColor(img_x1, cv2.COLOR_GRAY2BGR)

    for m in good:
        cv2.circle(img_x_bgr,  tuple(map(int, kp_x [m.queryIdx].pt)), 4, (0, 255, 0), -1)
        cv2.circle(img_x1_bgr, tuple(map(int, kp_x1[m.trainIdx].pt)), 4, (0, 255, 0), -1)

    combined = np.hstack([img_x_bgr, img_x1_bgr])
    offset   = img_x_bgr.shape[1]

    for m in good[:50]:
        pt_a = tuple(map(int, kp_x [m.queryIdx].pt))
        pt_b_raw = tuple(map(int, kp_x1[m.trainIdx].pt))
        pt_b = (pt_b_raw[0] + offset, pt_b_raw[1])
        cv2.line(combined, pt_a, pt_b, (0, 255, 0), 1)

    cv2.putText(combined, f'Frame {frame_x:06d} (left)',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(combined, f'Frame {frame_x1:06d} (left)',
                (offset + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(combined, f'Good matches: {len(good)}',
                (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    cv2.imshow(f'Consecutive left-camera matching (frame {frame_x} → {frame_x1})', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
