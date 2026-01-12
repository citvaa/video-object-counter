# Video Object Counter - HOG Descriptor Based Detection

## Overview
This project implements an object counter for video files using a combination of color-based filtering and HOG (Histogram of Oriented Gradients) descriptors. The system is designed to detect and count objects in video frames with robustness to variations in lighting and object appearance.

## Algorithm Overview

### 1. **Color-Based Filtering (HSV)**
   - Converts frames from BGR to HSV color space
   - Filters objects based on Value (brightness) and Saturation thresholds
   - Targets low-saturation objects (grayscale/white objects)
   - Parameters:
     - `v_th`: Value threshold (default: 185)
     - `s_th`: Saturation threshold (default: 70)

### 2. **Morphological Operations**
   - Applies closing operation to fill small holes
   - Applies opening operation to remove small noise
   - Uses elliptical 3x3 kernel for smooth operations

### 3. **Contour Detection & Filtering**
   - Finds external contours in the binary mask
   - Filters by:
     - **Area**: Between `min_area` (500 px²) and `max_area` (8000 px²)
     - **Aspect Ratio**: Between `ar_low` (0.7) and `ar_high` (1.3) for roughly square objects

### 4. **HOG Descriptor Verification**
   - Extracts region of interest (ROI) for each candidate contour
   - Computes HOG descriptor with 9 bins
   - Filters objects with low HOG norm (`hog_norm_thresh`: 2.0)
   - Ensures detected contours represent real objects

### 5. **Robust Count Estimation**
   - Computes per-frame object counts
   - Returns the 80th percentile of frame counts
   - Reduces impact of outliers and momentary noise

## File Structure
```
.
├── solution.py          # Main implementation
├── data/
│   └── count.csv        # Ground truth counts (video, count)
├── *.mp4                # Video files to process
└── README.md            # This file
```

## Requirements
- Python 3.7+
- OpenCV (cv2)
- NumPy
- CSV files with ground truth counts

## Installation

```bash
# Install required packages
pip install opencv-python numpy
```

## Usage

### Basic Usage
```bash
# Run with default data directory
python solution.py

# Run with custom data directory
python solution.py path/to/data
```

### Expected Data Format
The `count.csv` file should contain:
```
video,count
video1,150
video2,230
...
```

Videos should be named as `{video_name}.mp4` in the same directory as `count.csv`.

## Parameters

Key detection parameters in `main()`:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `v_th` | 185 | HSV value threshold |
| `s_th` | 70 | HSV saturation threshold |
| `min_area` | 500 | Minimum contour area (pixels²) |
| `max_area` | 8000 | Maximum contour area (pixels²) |
| `ar_low` | 0.7 | Minimum aspect ratio |
| `ar_high` | 1.3 | Maximum aspect ratio |
| `percentile` | 0.8 | Percentile for robust estimation |
| `hog_win` | (48, 48) | HOG window size |
| `hog_norm_thresh` | 2.0 | Minimum HOG norm |

## Output
The program computes and prints:
- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and ground truth counts
- Lower MAE indicates better performance

## Performance Notes
- The algorithm is tuned for detecting light-colored objects
- Morphological operations help with object connectivity
- HOG descriptor provides robustness to slight variations
- Percentile-based estimation is robust to frame-level noise

## Tuning Tips
- Adjust `v_th` and `s_th` if objects have different colors
- Modify `min_area` and `max_area` for different object sizes
- Change `ar_low` and `ar_high` for non-square objects
- Increase `hog_norm_thresh` to be more selective
- Adjust `percentile` to be more/less aggressive (0.5-0.9 recommended)

## Functions

### `get_hog(win_size)`
Initializes and returns a configured HOG descriptor.

### `load_true_counts(count_path)`
Loads ground truth counts from CSV file.

### `count_video(video_path, hog, params)`
Processes a single video and returns the estimated object count.

### `main()`
Main entry point: loads data, processes videos, evaluates MAE.

## Author
Vuk VIcentic  
Soft Computing - K2 Assignment
