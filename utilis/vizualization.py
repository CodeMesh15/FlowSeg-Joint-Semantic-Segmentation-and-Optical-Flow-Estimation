
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Using a standard color map, e.g., from Cityscapes
CITYSCAPES_COLOR_MAP = np.array([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
], dtype=np.uint8)

def mask_to_color(mask, color_map=CITYSCAPES_COLOR_MAP):
    """
    Converts a segmentation mask (with class IDs) to a color image.
    
    Args:
        mask (np.ndarray): A 2D array of integer class IDs.
        color_map (np.ndarray): A color map for each class ID.
        
    Returns:
        np.ndarray: A 3D array (color image).
    """
    # Create an empty RGB image
    color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Apply the color map
    for class_id, color in enumerate(color_map):
        color_mask[mask == class_id] = color
        
    return color_mask

def flow_to_color(flow):
    """
    Converts an optical flow field (u, v) to a color image using HSV color space.
    
    Args:
        flow (np.ndarray): A 3D array of shape (H, W, 2) representing the flow.
        
    Returns:
        np.ndarray: A 3D array (color image).
    """
    h, w = flow.shape[:2]
    
    # Calculate magnitude and angle of the 2D vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Create an HSV image
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2  # Hue (Angle)
    hsv[..., 1] = 255  # Saturation (Full)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value (Magnitude)
    
    # Convert HSV to BGR for display with OpenCV
    bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr_flow

def visualize_sample(image, pred_mask, gt_mask, pred_flow, gt_flow):
    """
    Creates a composite visualization of a single sample's inputs, predictions, and ground truths.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Convert tensors to numpy if needed and prepare for display
    # (Assuming inputs are numpy arrays for this function)
    image = np.asarray(image)

    # Row 1: Original Image, Ground Truths
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(mask_to_color(gt_mask))
    axes[0, 1].set_title("Ground Truth Mask")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(flow_to_color(gt_flow))
    axes[0, 2].set_title("Ground Truth Flow")
    axes[0, 2].axis('off')

    # Row 2: Predictions
    axes[1, 0].axis('off') # Empty space for alignment
    
    axes[1, 1].imshow(mask_to_color(pred_mask))
    axes[1, 1].set_title("Predicted Mask")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(flow_to_color(pred_flow))
    axes[1, 2].set_title("Predicted Flow")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
