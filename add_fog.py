import cv2
import numpy as np
import os

def add_fog(image, intensity=0.5):
    # Create fog effect
    H, W, C = image.shape
    fog = np.random.normal(loc=255, scale=intensity*255, size=(H, W, C)).astype(np.uint8)
    foggy_image = cv2.addWeighted(image, 1 - intensity, fog, intensity, 0)
    return foggy_image

def process_images(input_folder, foggy_folder, intensity=0.5):
    if not os.path.exists(foggy_folder):
        os.makedirs(foggy_folder)
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if os.path.isfile(img_path):
            image = cv2.imread(img_path)
            foggy_image = add_fog(image, intensity)
            foggy_output_path = os.path.join(foggy_folder, filename)
            cv2.imwrite(foggy_output_path, foggy_image)

# Process train, test, val folders
for folder in ['train', 'test', 'val']:
    input_folder = os.path.join('DIOR_dataset/images', folder)
    foggy_folder = os.path.join('DIOR_dataset/images', f'{folder}_foggy')
    process_images(input_folder, foggy_folder, intensity=0.5)
