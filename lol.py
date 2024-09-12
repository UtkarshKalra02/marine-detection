from PIL import Image
import os

images_path = "dataimg/Images"

for filename in os.listdir(images_path):
    if filename.endswith(".jpg"):
        image_path = os.path.join(images_path, filename)
        try:
            img = Image.open(image_path)
            img.verify()  # Check for corruption
        except (IOError, SyntaxError) as e:
            print(f"Corrupt image: {image_path}")