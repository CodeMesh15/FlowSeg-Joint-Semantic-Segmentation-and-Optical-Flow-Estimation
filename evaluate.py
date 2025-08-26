
import torch
from torch.utils.data import DataLoader
import argparse
import numpy as np
from tqdm import tqdm

from data.custom_dataset import KittiFlowSegDataset
from models.flowseg_model import FlowSegModel
from utils.metrics import calculate_miou, calculate_aepe

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading (for the validation/test set)
    test_dataset = KittiFlowSegDataset(
        image_dir=args.image_dir, 
        flow_dir=args.flow_dir, 
        seg_dir=args.seg_dir
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 2. Model Loading
    model = FlowSegModel(num_seg_classes=19).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval() # Set model to evaluation mode

    total_miou = 0.0
    total_aepe = 0.0
    num_samples = 0

    print("--- Starting Evaluation ---")
    with torch.no_grad(): # No need to calculate gradients
        for batch in tqdm(test_dataloader):
            # Move data to device
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)
            gt_flow = batch['flow'] # Keep on CPU as numpy for metrics
            gt_seg_mask = batch['segmentation']

            # Forward pass
            pred_seg_logits, pred_flow = model(image1, image2)
            
            # Process segmentation output
            pred_seg_masks = torch.argmax(pred_seg_logits, dim=1).cpu().numpy()
            
            # Process flow output
            pred_flow = pred_flow.cpu().numpy()
            
            # Calculate metrics for each item in the batch
            for i in range(len(pred_seg_masks)):
                total_miou += calculate_miou(pred_seg_masks[i], gt_seg_mask[i].numpy(), num_classes=19)
                total_aepe += calculate_aepe(pred_flow[i], gt_flow[i].numpy())
                num_samples += 1

    # Calculate average metrics
    avg_miou = total_miou / num_samples
    avg_aepe = total_aepe / num_samples

    print("--- Evaluation Complete ---")
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average AEPE: {avg_aepe:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the FlowSeg model.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model .pth file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Path to test image directory.')
    parser.add_argument('--flow_dir', type=str, required=True, help='Path to test flow directory.')
    parser.add_argument('--seg_dir', type=str, required=True, help='Path to test segmentation directory.')
    parser.add_argument('--batch_size', type=int, default=8)

    args = parser.parse_args()
    evaluate(args)
