# FlowSeg-Joint-Semantic-Segmentation-and-Optical-Flow-Estimation

# 1. Project Overview
- This project aims to build a deep learning model that can simultaneously perform semantic segmentation (identifying and categorizing objects in each frame) and optical flow estimation (estimating the motion of pixels between frames). The core idea is that these two tasks are complementary; understanding motion can help delineate objects, and understanding object boundaries can improve motion estimation. This implementation is based on the "FlowSeg" project, which focused on using novel Deep CNN architectures to jointly solve these problems.


# 2. Core Objectives
- To build a deep convolutional neural network capable of joint, multi-task learning.

- To simultaneously estimate semantic segmentation masks and optical flow fields from video sequences.

- To implement a training and evaluation pipeline for this dual-task problem.
