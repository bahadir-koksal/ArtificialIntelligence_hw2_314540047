import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from PIL import Image

# This class wraps the standard CIFAR10 dataset to return two augmentations
class CIFAR10Pair(CIFAR10):
    """
    CIFAR10 Dataset that returns two differently augmented versions of the same image.
    """
    def __getitem__(self, index):
        # Get the original image and target label from the base CIFAR10 class
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        # Apply the same transformation pipeline twice to get two different views
        # Because the transforms are stochastic (random), pos_1 and pos_2 will be different
        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target

def get_train_transform():
    """
    Returns the SimCLR augmentation pipeline as described in the assignment.
    """
    return transforms.Compose([
        # 1. Random Crop
        transforms.RandomResizedCrop(32),
        # 2. Horizontal Flip
        transforms.RandomHorizontalFlip(p=0.5),
        # 3. Color Jitter
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        # 4. Random Grayscale
        transforms.RandomGrayscale(p=0.2),
        # 5. Conversion to Tensor
        transforms.ToTensor(),
        # 6. Normalization Required step)(
        # Standard CIFAR-10 mean and std values
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def get_test_transform():
    """
    Simple transform for kNN monitor and evaluation (no random augmentations).
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])