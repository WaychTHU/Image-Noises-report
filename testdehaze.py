import cv2
import numpy as np
import os

def get_dark_channel(image, size=15):
    """Get the dark channel prior in the (RGB) image data."""
    min_channel = np.amin(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def get_atmosphere(image, dark_channel, percentile=0.001):
    """Estimate the atmosphere light in the image."""
    flat_image = image.reshape(-1, 3)
    flat_dark = dark_channel.ravel()
    search_idx = (-flat_dark).argsort()[:int(flat_dark.size * percentile)]
    atmosphere = np.mean(flat_image[search_idx], axis=0)
    return atmosphere

def get_transmission(image, atmosphere, omega=0.95, size=15):
    """Estimate the transmission map."""
    normalized_image = image / atmosphere
    transmission = 1 - omega * get_dark_channel(normalized_image, size)
    return transmission

def recover_image(image, transmission, atmosphere, t0=0.1):
    """Recover the scene radiance."""
    transmission = np.clip(transmission, t0, 1)
    recovered = (image - atmosphere) / transmission[:, :, np.newaxis] + atmosphere
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return recovered

def dehaze_image(image_path, output_path):
    image = cv2.imread(image_path)
    dark_channel = get_dark_channel(image)
    atmosphere = get_atmosphere(image, dark_channel)
    transmission = get_transmission(image, atmosphere)
    recovered_image = recover_image(image, transmission, atmosphere)
    cv2.imwrite(output_path, recovered_image)

# 定义输入和输出文件夹
dataset_path = 'DIOR_dataset/images'
folders = ['train_foggy', 'test_foggy', 'val_foggy']
for folder in folders:
    input_folder = os.path.join(dataset_path, folder)
    output_folder = os.path.join(dataset_path, folder.replace('foggy', 'dehazed'))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# 处理所有图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            dehaze_image(input_path, output_path)

print("去雾处理完成！")
