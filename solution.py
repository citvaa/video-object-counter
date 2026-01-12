"""
Video Object Counter using HOG Descriptor and Color-based Detection

This module detects and counts objects in video frames using:
- HSV color space filtering for object localization
- Morphological operations for noise reduction
- Contour detection with area and aspect ratio filtering
- HOG (Histogram of Oriented Gradients) descriptor for object verification
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np


def get_hog(win_size):
    """
    Initialize HOG (Histogram of Oriented Gradients) descriptor.
    
    Args:
        win_size (tuple): (height, width) of the window to analyze
        
    Returns:
        cv2.HOGDescriptor: Configured HOG descriptor object
    """
    nbins = 9  # Number of histogram bins
    cell_size = (8, 8)  # Size of each cell in pixels
    block_size = (3, 3)  # Number of cells per block
    hog = cv2.HOGDescriptor(
        _winSize=(win_size[1] // cell_size[1] * cell_size[1],
                  win_size[0] // cell_size[0] * cell_size[0]),
        _blockSize=(block_size[1] * cell_size[1],
                    block_size[0] * cell_size[0]),
        _blockStride=(cell_size[1], cell_size[0]),
        _cellSize=(cell_size[1], cell_size[0]),
        _nbins=nbins,
    )
    return hog


def load_true_counts(count_path):
    """
    Load ground truth object counts from CSV file.
    
    Args:
        count_path (Path): Path to CSV file with 'video' and 'count' columns
        
    Returns:
        dict: Mapping of video names to their true object counts
    """
    with count_path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["video"]: int(row["count"]) for row in reader}


def count_video(video_path, hog, params):
    """
    Count objects in a video file using color filtering and HOG descriptor.
    
    Args:
        video_path (Path): Path to the video file
        hog (cv2.HOGDescriptor): HOG descriptor for object verification
        params (dict): Dictionary containing detection parameters:
            - v_th: Value threshold for HSV filtering
            - s_th: Saturation threshold for HSV filtering
            - min_area/max_area: Contour area bounds
            - ar_low/ar_high: Aspect ratio bounds
            - kernel: Morphological kernel
            - percentile: Percentile of frame counts to return
            - hog_win: HOG window size
            - hog_norm_thresh: HOG norm threshold
    
    Returns:
        int: Estimated count of objects based on percentile of frame counts
    """
    cap = cv2.VideoCapture(str(video_path))
    counts = []  # List of per-frame object counts
    frames_read = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV color space for better color-based filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Create mask: filter for low-saturation objects (white/light objects)
        mask = cv2.inRange(
            hsv,
            (0, 0, params["v_th"]),
            (180, params["s_th"], 255),
        )

        # Remove noise: close small holes, open small objects
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, params["kernel"], iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, params["kernel"], iterations=1)

        # Find contours of connected components
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = 0
        for c in contours:
            # Filter by area: skip contours too small or too large
            area = cv2.contourArea(c)
            if area < params["min_area"] or area > params["max_area"]:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            # Filter by aspect ratio: keep roughly square objects
            ar = w / h
            if not (params["ar_low"] <= ar <= params["ar_high"]):
                continue

            # Extract ROI and compute HOG descriptor for verification
            roi = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(roi, params["hog_win"], interpolation=cv2.INTER_AREA)
            feat = hog.compute(roi)
            # Skip if HOG norm is too low (likely not a real object)
            if np.linalg.norm(feat) < params["hog_norm_thresh"]:
                continue

            cnt += 1

        counts.append(cnt)
        frames_read += 1

    cap.release()

    if not counts:
        return 0

    # Return percentile of frame counts (robust to outliers)
    counts_sorted = sorted(counts)
    idx = int(len(counts_sorted) * params["percentile"])
    if idx >= len(counts_sorted):
        idx = len(counts_sorted) - 1
    return counts_sorted[idx]


def main():
    """
    Main function: load ground truth, process videos, and evaluate performance.
    """
    # Get data directory from command line or default to 'data'
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    # Get data directory from command line or default to 'data'
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    count_path = data_dir / "count.csv"
    true_counts = load_true_counts(count_path)

    # Detection parameters (tuned for the specific use case)
    params = {
        "v_th": 185,  # Value threshold for HSV (darker objects)
        "s_th": 70,  # Saturation threshold (low saturation = grayscale objects)
        "min_area": 500,  # Minimum contour area in pixels
        "max_area": 8000,  # Maximum contour area in pixels
        "ar_low": 0.7,  # Aspect ratio lower bound
        "ar_high": 1.3,  # Aspect ratio upper bound
        "kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        "nframes": 40,
        "percentile": 0.8,  # Return 80th percentile of frame counts
        "hog_win": (48, 48),  # HOG window size
        "hog_norm_thresh": 2.0,  # Minimum HOG norm to accept detection
    }

    hog = get_hog(params["hog_win"])

    # Process each video and get predictions
    preds = {}
    for video_name in true_counts.keys():
        video_path = data_dir / f"{video_name}.mp4"
        preds[video_name] = count_video(video_path, hog, params)

    # Calculate and display Mean Absolute Error
    mae = sum(abs(preds[name] - true_counts[name]) for name in true_counts) / len(true_counts)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")


if __name__ == "__main__":
    main()
