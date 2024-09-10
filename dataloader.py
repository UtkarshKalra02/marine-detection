# Import necessary libraries
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Define paths for images and masks
images_path = "dataimg/Images"
masks_path = "dataimg/Images"
preprocessed_images_path = "dataimg/PreProcessedImages"
preprocessed_masks_path = "dataimg/PreProcessedMasks"

# Create directories if they do not exist
os.makedirs(preprocessed_images_path, exist_ok=True)
os.makedirs(preprocessed_masks_path, exist_ok=True)

# Define preprocessing transforms for images
preprocess_image = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize image to a fixed size (256x256 in this case)
    transforms.ToTensor(),           # Convert image to a torch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   # Normalize the image using ImageNet means and std
                         std=[0.229, 0.224, 0.225])
])

# Define preprocessing for masks (no normalization, but resizing)
preprocess_mask = transforms.Compose([
    transforms.Resize((256, 256)),   # Resize mask to match the image
    transforms.ToTensor()            # Convert mask to a tensor
])

# Function to preprocess individual image and mask
def preprocess_image_and_mask(image_path, mask_path, image_save_path, mask_save_path):
    # Load the image and mask
    image = Image.open(image_path).convert("RGB")  # Ensure image is in RGB mode
    mask = Image.open(mask_path).convert("L")      # Ensure mask is in grayscale mode (L)

    # Apply preprocessing transforms
    image = preprocess_image(image)
    mask = preprocess_mask(mask)

    # Save the preprocessed image and mask
    torch.save(image, image_save_path)
    torch.save(mask, mask_save_path)

    return image, mask

# Preprocess all images and masks in the folder
for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_path, filename)
        mask_path = os.path.join(masks_path, filename)
        image_save_path = os.path.join(preprocessed_images_path, filename.replace(".jpg", ".pt"))
        mask_save_path = os.path.join(preprocessed_masks_path, filename.replace(".jpg", ".pt"))

        image, mask = preprocess_image_and_mask(image_path, mask_path, image_save_path, mask_save_path)

        # Visualize an example image and mask
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Preprocessed Image')
        plt.imshow(image.permute(1, 2, 0).numpy())
        plt.subplot(1, 2, 2)
        plt.title('Preprocessed Mask')
        plt.imshow(mask[0], cmap='gray')
        plt.show()