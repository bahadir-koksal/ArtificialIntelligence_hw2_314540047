import torch
import torch.nn as nn
import torchvision.models as models

class SimCLRBackbone(nn.Module):
    """
    Modified ResNet-18 for 32x32 CIFAR-10 images.
    """
    def __init__(self):
        super(SimCLRBackbone, self).__init__()
        
        # Load standard ResNet-18 with NO pretrained weights
        resnet = models.resnet18(weights=None)
        
        # Modify the first convolution for 32x32 images
        # Changing from 7x7 to 3x3, stride=1, padding=1
        resnet.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Replace the max pooling layer with an Identity layer
        resnet.maxpool = nn.Identity()
        
        # Strip the final fully connected classification layer
        # We only want the features up to the adaptive average pooling
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

    def forward(self, x):
        # Output shape from backbone: (Batch, 512, 1, 1)
        x = self.backbone(x)
        # Flatten to a 1D vector of 512 elements per image: (Batch, 512)
        h = torch.flatten(x, 1)
        return h

class SimCLRModel(nn.Module):
    """
    The complete SimCLR network: Backbone + Projector Head.
    """
    def __init__(self):
        super(SimCLRModel, self).__init__()
        
        # Initialize the modified ResNet-18
        self.backbone = SimCLRBackbone()
        
        # Initialize the Projector Head (512 -> 512 hidden -> 128)
        self.projector = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512), # Standard practice for SimCLR MLPs
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        # Get the 512-dim representation from the backbone
        h = self.backbone(x)
        
        # Pass it through the projector head to get the 128-dim output
        z = self.projector(h)
        # We return BOTH 'h' and 'z'
        # 'h' will be used for your kNN monitor
        # 'z' will be used for calculating the NT-Xent loss
        return h, z