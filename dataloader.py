import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as F

# Constants for band normalization (if working with satellite images, adjust accordingly)
class_distr = torch.Tensor([
    0.00452, 0.00203, 0.00254, 0.00168, 0.00766, 0.15206, 0.20232,
    0.35941, 0.00109, 0.20218, 0.03226, 0.00693, 0.01322, 0.01158, 0.00052
])

# Mean and standard deviation values for RGB images
bands_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
bands_std = np.array([0.229, 0.224, 0.225], dtype='float32')


# Custom rotation transformation
class RandomRotationTransform:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = float(np.random.choice(self.angles))  # Ensure angle is float
        return F.rotate(x, angle)

# Function to generate weights for classes
def gen_weights(class_distr, c=1.0):
    class_weights = [c / cls if cls != 0 else 0 for cls in class_distr]
    return torch.FloatTensor(class_weights)


# Dataset class for GenDEBRIS
class GenDEBRIS(Dataset):
    def __init__(self, split, root_dir, transform=None, standardization=None, agg_to_water=True):
        """
        Dataset class for loading satellite images and corresponding masks (e.g., debris segmentation).

        Args:
        - split: 'train' or 'val', defining which split to use.
        - root_dir: Path to the directory containing the images and masks.
        - transform: Optional transformations to apply to the images and masks.
        - standardization: Normalization applied to the bands, typically per-channel mean and std.
        - agg_to_water: Boolean flag to aggregate debris classes into water, specific to your dataset.
        """
        self.split = split
        self.root_dir = root_dir
        self.transform = transform
        self.standardization = standardization
        self.agg_to_water = agg_to_water

        # Get the list of images and masks
        if self.split == 'train':
            self.image_dir = os.path.join(self.root_dir, "Images")
            self.mask_dir = os.path.join(self.root_dir, "Masks")
        elif self.split == 'val':
            self.image_dir = os.path.join(self.root_dir, "Images")
            self.mask_dir = os.path.join(self.root_dir, "Masks")

        self.images_list = sorted(os.listdir(self.image_dir))
        self.masks_list = sorted(os.listdir(self.mask_dir))

        assert len(self.images_list) == len(self.masks_list), "Number of images and masks must match!"

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images_list[idx])
        mask_path = os.path.join(self.mask_dir, self.masks_list[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure the mask is loaded as a single-channel image

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if self.standardization:
            image = self.standardization(image)

        if self.agg_to_water:
            mask = self.aggregate_classes(mask)

        mask = torch.squeeze(mask, dim=0)  # Remove the extra channel dimension, if present

        return image, mask

    def aggregate_classes(self, mask):
        """
        This method is used to aggregate specific classes into the "water" class.
        Adjust the logic based on the class distribution of your dataset.
        """
        mask_np = np.array(mask)
        # For example, if class indices from 6 to 10 should be aggregated into class 6 (water)
        agg_indices = [7, 8, 9, 10]  # Adjust based on actual classes
        mask_np[mask_np > 6] = 6  # Aggregate other classes into class 6 (water)
        return torch.from_numpy(mask_np).long()


# Example usage:
if __name__ == "__main__":
    root_dir = "dataimg"  # Root directory containing "train" and "val" folders
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        RandomRotationTransform([-90, 0, 90, 180]),  # Custom random rotation
        transforms.RandomHorizontalFlip(),
    ])
    standardization = transforms.Normalize(bands_mean, bands_std)

    # Initialize dataset and dataloader
    train_dataset = GenDEBRIS(split='train', root_dir=root_dir, transform=transform, standardization=standardization)
    val_dataset = GenDEBRIS(split='val', root_dir=root_dir, transform=transform, standardization=standardization)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Loop over dataset
    for images, masks in train_loader:
        print(f"Batch of images shape: {images.shape}")
        print(f"Batch of masks shape: {masks.shape}")