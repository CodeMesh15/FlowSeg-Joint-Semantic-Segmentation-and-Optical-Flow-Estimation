
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from data.custom_dataset import KittiFlowSegDataset # Assuming your data is prepared
from models.flowseg_model import FlowSegModel

def train(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    # NOTE: You need to implement the actual data transforms (resizing, normalization, etc.)
    dataset = KittiFlowSegDataset(
        image_dir=args.image_dir, 
        flow_dir=args.flow_dir, 
        seg_dir=args.seg_dir
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 2. Model Initialization
    model = FlowSegModel(num_seg_classes=19).to(device) # 19 classes for Cityscapes/KITTI

    # 3. Loss Functions
    # Use CrossEntropyLoss for the multi-class segmentation task
    loss_segmentation_fn = nn.CrossEntropyLoss()
    # Use L1 or MSE loss for the regression task of optical flow
    loss_flow_fn = nn.MSELoss() 

    # 4. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    print("--- Starting Training ---")
    for epoch in range(args.epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            # Move data to the selected device
            image1 = batch['image1'].to(device)
            image2 = batch['image2'].to(device)
            gt_flow = batch['flow'].to(device)
            gt_seg_mask = batch['segmentation'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred_seg, pred_flow = model(image1, image2)
            
            # Calculate losses
            loss_seg = loss_segmentation_fn(pred_seg, gt_seg_mask)
            loss_flow = loss_flow_fn(pred_flow, gt_flow)
            
            # Combined loss (you can tune the weights alpha and beta)
            alpha = 0.8
            beta = 0.2
            combined_loss = (alpha * loss_seg) + (beta * loss_flow)
            
            # Backward pass and optimize
            combined_loss.backward()
            optimizer.step()

            running_loss += combined_loss.item()
            if i % 10 == 9: # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print("--- Finished Training ---")
    
    # Save the trained model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the FlowSeg model.")
    # Add arguments for data paths, learning rate, epochs, etc.
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--flow_dir', type=str, required=True)
    parser.add_argument('--seg_dir', type=str, required=True)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--save_path', type=str, default='flowseg_model.pth')
    
    args = parser.parse_args()
    train(args)
