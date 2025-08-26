
import numpy as np

def calculate_aepe(predicted_flow, ground_truth_flow):
    """
    Calculates the Average Endpoint Error (AEPE) between two flow fields.

    Args:
        predicted_flow (np.ndarray): The predicted flow field, shape (H, W, 2).
        ground_truth_flow (np.ndarray): The ground truth flow field, shape (H, W, 2).

    Returns:
        float: The average endpoint error.
    """
    # Ensure inputs are numpy arrays
    predicted_flow = np.asarray(predicted_flow)
    ground_truth_flow = np.asarray(ground_truth_flow)

    # Calculate the squared difference for each component (u, v)
    squared_diff = (predicted_flow - ground_truth_flow) ** 2

    # Sum the squared differences and take the square root for Euclidean distance
    endpoint_error = np.sqrt(np.sum(squared_diff, axis=2))

    # Return the average error across all pixels
    return np.mean(endpoint_error)


def calculate_miou(predicted_mask, ground_truth_mask, num_classes):
    """
    Calculates the Mean Intersection-over-Union (mIoU) for semantic segmentation.

    Args:
        predicted_mask (np.ndarray): The predicted segmentation mask, shape (H, W).
        ground_truth_mask (np.ndarray): The ground truth segmentation mask, shape (H, W).
        num_classes (int): The total number of classes.

    Returns:
        float: The mean IoU score.
    """
    # Ensure inputs are numpy arrays
    predicted_mask = np.asarray(predicted_mask)
    ground_truth_mask = np.asarray(ground_truth_mask)
    
    iou_list = []
    
    for class_id in range(num_classes):
        # Create boolean masks for the current class
        pred_is_class = (predicted_mask == class_id)
        gt_is_class = (ground_truth_mask == class_id)
        
        # Calculate intersection and union
        intersection = np.sum(pred_is_class & gt_is_class)
        union = np.sum(pred_is_class | gt_is_class)
        
        if union == 0:
            # If a class is not present in either mask, its IoU is often considered 0 or ignored.
            # We'll add 0 to our list, but this can be handled differently.
            iou = 0.0
        else:
            iou = intersection / union
            
        iou_list.append(iou)
        
    # Return the mean of IoUs across all classes
    return np.mean(iou_list)
