import cv2
import os
import glob
import numpy as np

# Path to KITTI dataset images (update this to your actual dataset path)
DATASET_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'dataset', 'dataset')

# KITTI odometry sequence path, e.g. dataset/sequences/00/
DATA_PATH = os.path.join(DATASET_PATH, 'sequences', '00') + os.sep

def read_images(idx):
    img_name = '{:06d}.png'.format(idx)
    img1 = cv2.imread(DATA_PATH + 'image_0' + os.sep + img_name, 0)
    img2 = cv2.imread(DATA_PATH + 'image_1' + os.sep + img_name, 0)
    return img1, img2

def get_image_paths(dataset_path):
    extensions = ('*.png', '*.jpg', '*.jpeg')
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(dataset_path, '**', ext), recursive=True))
    return sorted(paths)

def run_sift(img):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_keypoints(img1, kp1, des1, img2, kp2, des2, ratio=0.9, use_significance_test=True):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    discarded_matches = []
    for m, n in matches:
        passes_significance = (not use_significance_test) or (m.distance < ratio * n.distance)
        if passes_significance:
            good_matches.append(m)
        else:
            discarded_matches.append(m)

    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return good_matches, discarded_matches, img_matches


def find_correct_discarded(kp1, kp2, discarded, epipolar_thresh=1.0):
    """Return first discarded match that satisfies the stereo epipolar constraint."""
    for m in discarded:
        pt1 = kp1[m.queryIdx].pt
        pt2 = kp2[m.trainIdx].pt
        if abs(pt1[1] - pt2[1]) < epipolar_thresh:
            return m
    return None

if __name__ == '__main__':
    num_frames = len(glob.glob(DATA_PATH + 'image_0' + os.sep + '*.png'))
    if num_frames == 0:
        print(f"No frames found at {DATA_PATH}. Check your DATA_PATH.")
        exit(1)

    print(f"Sequence 00: {num_frames} frames found.")

    for idx in range(num_frames - 1):
        img1, img2 = read_images(idx)
        if img1 is None or img2 is None:
            print(f"Could not load frame {idx}, skipping.")
            continue

        kp1, des1 = run_sift(img1)
        kp2, des2 = run_sift(img2)
        print(f"Frame {idx:06d}: {len(kp1)} / {len(kp2)} keypoints (left / right)")

        ratio = 0.7
        good_matches, discarded_matches, img_matches = match_keypoints(img1, kp1, des1, img2, kp2, des2, ratio=ratio)
        print(f"  Good matches: {len(good_matches)}, Discarded matches: {len(discarded_matches)}")

        # Find a discarded match that is actually correct (satisfies epipolar constraint).
        # If none found at the current ratio, increase ratio (stricter test → more discarded)
        # until we find one.
        correct_discarded = find_correct_discarded(kp1, kp2, discarded_matches)
        used_ratio = ratio
        while correct_discarded is None and used_ratio < 0.99:
            used_ratio = round(used_ratio + 0.05, 2)
            _, discarded_matches_new, _ = match_keypoints(img1, kp1, des1, img2, kp2, des2, ratio=used_ratio)
            correct_discarded = find_correct_discarded(kp1, kp2, discarded_matches_new)
            if correct_discarded is not None:
                print(f"  (ratio increased to {used_ratio:.2f} to find a correct discarded match)")

        if correct_discarded is not None:
            pt1 = tuple(map(int, kp1[correct_discarded.queryIdx].pt))
            pt2 = tuple(map(int, kp2[correct_discarded.trainIdx].pt))

            img1_dot = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
            img2_dot = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

            # Draw circles on each image first, then hstack so lines can cross the boundary
            for m in good_matches[:500]:
                gpt1 = tuple(map(int, kp1[m.queryIdx].pt))
                gpt2 = tuple(map(int, kp2[m.trainIdx].pt))
                cv2.circle(img1_dot, gpt1, 5, (0, 255, 0), -1)
                cv2.circle(img2_dot, gpt2, 5, (0, 255, 0), -1)

            cv2.circle(img1_dot, pt1, 8, (0, 0, 255), -1)
            cv2.circle(img2_dot, pt2, 8, (0, 0, 255), -1)
            cv2.putText(img1_dot, 'Left', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img2_dot, 'Right', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            combined = np.hstack([img1_dot, img2_dot])
            offset = img1_dot.shape[1]

            # Draw lines connecting each good match across the combined image
            for m in good_matches[:20]:
                gpt1 = tuple(map(int, kp1[m.queryIdx].pt))
                gpt2 = (tuple(map(int, kp2[m.trainIdx].pt))[0] + offset, tuple(map(int, kp2[m.trainIdx].pt))[1])
                cv2.line(combined, gpt1, gpt2, (0, 255, 0), 1)

            # Line for the discarded-but-correct match
            pt2_combined = (pt2[0] + offset, pt2[1])
            cv2.line(combined, pt1, pt2_combined, (0, 0, 255), 1)
            cv2.putText(combined, f'Correct match discarded by ratio test (ratio={used_ratio:.2f})',
                        (10, combined.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow('Discarded Correct Match', combined)
            cv2.waitKey(0)
        else:
            print("  No correct discarded match found for this frame.")

        cv2.imshow('SIFT Feature Matching', img_matches)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
