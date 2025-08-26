# Required Datasets for FlowSeg Project

This document outlines the datasets required to train and evaluate the FlowSeg model. Due to the large size of the data, it is recommended to download them in the order specified.

---

### 1. Raw Image Sequences (KITTI Raw Data)

These are the input video frames for the model. The full dataset is over 100 GB, so it is highly recommended to start with a few sample drives.

-   **Source:** [KITTI Raw Data Website](http://www.cvlibs.net/datasets/kitti/raw_data.php)
-   **Instructions:** Navigate to the website and download a few of the "synced+rectified" data zip files. The 'Residential' category is a good place to start.
-   **Example Files to Download:**
    -   `2011_09_26_drive_0001_sync.zip` (13 MB)
    -   `2011_09_26_drive_0002_sync.zip` (12 MB)
    -   `2011_09_26_drive_0005_sync.zip` (14 MB)
    -   `2011_09_26_drive_0009_sync.zip` (13 MB)

---

### 2. Optical Flow Ground Truth (KITTI 2015 Benchmark)

This dataset provides the ground truth optical flow annotations corresponding to the raw image sequences. These are essential for training the flow estimation head of the model.

-   **Source:** [KITTI 2015 Scene Flow/Optical Flow Benchmark](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
-   **File to Download:**
    -   `data_scene_flow.zip` (4 GB) - This contains the training data with ground truth for optical flow.

---

### 3. Semantic Segmentation Ground Truth (SemanticKITTI)

This dataset provides dense, point-wise semantic labels for the KITTI sequences, which are required for training the semantic segmentation head of the model.

-   **Source:** [SemanticKITTI Dataset Website](http://semantic-kitti.org/dataset.html)
-   **File to Download:**
    -   `semantic-kitti.zip` (25 GB) - This file contains the semantic and instance labels for the odometry benchmark sequences 00-10, which are needed for training.

---

### Next Steps

After downloading these files, you will need to:
1.  Unzip them into a common `/data` directory.
2.  Implement preprocessing scripts to align the raw images from a specific drive with their corresponding optical flow and semantic segmentation labels.
