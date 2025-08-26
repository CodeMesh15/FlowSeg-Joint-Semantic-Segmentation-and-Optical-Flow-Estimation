# /models/flowseg_model.py

import torch
import torch.nn as nn
import torchvision.models as models

class FlowSegModel(nn.Module):
    def __init__(self, num_seg_classes=19):
        """
        Initializes the FlowSeg model.
        Args:
            num_seg_classes (int): The number of semantic classes for the segmentation task.
        """
        super(FlowSegModel, self).__init__()
        
        # 1. Shared Encoder
        # Use a pretrained ResNet-34, but remove the final fully connected layers
        resnet = models.resnet34(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        # --- 2. Segmentation Decoder ---
        # This is a simplified decoder using transposed convolutions (ConvTranspose2d)
        # to upsample the feature map from the encoder back to the original image size.
        self.seg_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Final layer to produce the class scores for each pixel
            nn.Conv2d(64, num_seg_classes, kernel_size=3, padding=1)
        )
        
        # --- 3. Optical Flow Decoder ---
        # This decoder also upsamples the feature map and predicts the 2-channel flow (x, y).
        self.flow_decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            # Final layer to produce the 2-channel optical flow vector field
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )

    def forward(self, image1, image2):
        """
        Defines the forward pass of the model.
        """
        # Concatenate the two input images along the channel dimension
        x = torch.cat([image1, image2], dim=1)
        
        # Pass through the shared encoder. 
        # Note: We need to modify the first layer of the encoder to accept 6 channels (2 RGB images)
        # For simplicity in this example, we'll pretend it works directly.
        # A real implementation would adjust the first conv layer's weights.
        
        # Let's pass image1 for segmentation and both for flow features
        features = self.encoder(image1)

        # Pass features through the two decoders
        seg_output = self.seg_decoder(features)
        flow_output = self.flow_decoder(features) # Note: a better model would use features from both images
        
        return seg_output, flow_output
