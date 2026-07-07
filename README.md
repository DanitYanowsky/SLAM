# 🚗 Stereo Visual Odometry & SLAM

> A complete visual SLAM pipeline built from scratch — from raw stereo images to a globally optimized 3D trajectory.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white)
![GTSAM](https://img.shields.io/badge/GTSAM-Factor%20Graphs-orange)
![Dataset](https://img.shields.io/badge/Dataset-KITTI%20Odometry-lightgrey)
![Course](https://img.shields.io/badge/Course-Vision%20Aided%20Navigation%20%7C%20HUJI-purple)

---

## About

This project implements a full **stereo visual odometry and SLAM system** based on the *Vision Aided Navigation* course at the **Hebrew University of Jerusalem (HUJI)**.

Starting from calibrated stereo image pairs (KITTI benchmark), the system progressively builds up:

- Feature detection and stereo matching
- Metric 3D triangulation
- Robust frame-to-frame pose estimation (PnP + RANSAC)
- Multi-frame feature tracking
- Windowed Bundle Adjustment
- Global Pose Graph Optimization

The result is a full localization pipeline comparable to real-world autonomous driving systems.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        Stereo Images (KITTI)                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 1 · SIFT Detection & Matching  │
              │   Ratio test · Epipolar filtering   │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 2 · Stereo Triangulation       │
              │   Custom SVD · OpenCV validation    │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 3 · Visual Odometry            │
              │   PnP · RANSAC · 4-view quads       │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 4 · Feature Tracking Database  │
              │   Cross-frame track management      │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 5 · Bundle Adjustment          │
              │   GTSAM · Stereo factors · Windows  │
              └─────────────────┬──────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Ex 6 · Pose Graph Optimization    │
              │   GTSAM · Global trajectory         │
              └─────────────────┬──────────────────┘
                                │
                        ✅ Optimized 3D Trajectory
```

---

## What's Inside

### Ex 1 — SIFT Feature Detection & Stereo Matching
**`code/feature_matching/sift_features.py`**

Detects and matches SIFT features across left/right stereo pairs using FLANN with Lowe's ratio test. Includes an analysis of the epipolar geometry to identify matches that are geometrically correct but discarded by the ratio test — motivating a deeper look at descriptor-based filtering.

---

### Ex 2 — Stereo Triangulation
**`code/triangulation/triangulation.py`**

Implements metric 3D reconstruction from stereo correspondences. The core is a hand-written **DLT triangulation via SVD**, validated numerically against OpenCV's built-in `triangulatePoints`. Outputs colored 3D point clouds across multiple frames.

---

### Ex 3 — Visual Odometry with PnP + RANSAC
**`code/visual_odometry/`**

Frame-to-frame pose estimation using **4-view quadruple correspondences** — features tracked simultaneously across left₀, right₀, left₁, right₁. Pose is solved with `solvePnP` (SQPNP) inside a RANSAC loop and refined by re-fitting on all inliers. Visualizes supporter/non-supporter splits to validate the estimated motion.

---

### Ex 4 — Feature Tracking Database
**`code/tracking/tracking_database.py`**

A custom `TrackingDB` data structure that accumulates feature **tracks** across the full video sequence. Each track links a 3D landmark to its stereo observations (`x_left`, `x_right`, `y`) over many frames. Supports efficient lookup by frame or track ID, and serializes the full database to disk.

---

### Ex 5 — Bundle Adjustment
**`code/bundle_adjustment/BundleAdjustment.py`**

Windowed **Bundle Adjustment** using GTSAM's `StereoCamera` factor graph. Jointly optimizes camera poses and 3D landmark positions over sliding windows of frames. Computes **marginal covariances** on the relative pose between each window's endpoints — feeding directly into the pose graph.

---

### Ex 6 — Pose Graph Optimization
**`code/pose_graph/PoseGraphOptimization.py`**

Closes the loop on the full trajectory. Builds a **pose graph** from the per-window relative poses and their covariances, then runs a global GTSAM optimization to produce a drift-corrected trajectory. Results are cached to avoid re-running the full Bundle Adjustment on repeated runs.

---

## Tech Stack

| Technology | Role |
|---|---|
| **Python 3** | Core language |
| **OpenCV** | Feature detection, PnP, triangulation |
| **NumPy** | Linear algebra, SVD, projective geometry |
| **GTSAM** | Factor graph optimization (BA + pose graph) |
| **Matplotlib** | 3D visualization and trajectory plots |
| **KITTI Dataset** | Stereo image sequences + ground truth poses |

---

## Dataset Setup

Download the [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) (sequence 00) and place it under `dataset/`:

```
dataset/
└── dataset/
    ├── sequences/
    │   └── 00/
    │       ├── image_0/        ← left camera frames  (.png)
    │       ├── image_1/        ← right camera frames (.png)
    │       └── calib.txt       ← stereo calibration
    └── poses/
        └── 00.txt              ← ground truth trajectory
```

---

## Getting Started

```bash
# Install dependencies
pip install opencv-python numpy matplotlib gtsam

# Run any module independently
python code/feature_matching/sift_features.py
python code/triangulation/triangulation.py
python code/visual_odometry/ransac_pnp.py
python code/tracking/ex4_tracking.py
python code/bundle_adjustment/BundleAdjustment.py
python code/pose_graph/PoseGraphOptimization.py
```

---

## Course

This project was developed as part of the **Vision Aided Navigation** course
at the **Hebrew University of Jerusalem (HUJI)**.

The course covers the mathematical foundations and practical implementation of
camera-based localization and mapping — from projective geometry and multi-view
reconstruction to full SLAM systems.
