import os
from PIL import Image
import numpy as np

def create_binary_mask(image_path, output_path, alpha_threshold=10):
    """
    将 RGBA 图像转换为二值黑白掩码图（用于 U^2-Net 训练）
    
    :param image_path: 输入图像路径（PNG）
    :param output_path: 输出掩码图路径（PNG）
    :param alpha_threshold: alpha 通道阈值（小于该值为背景）
    """
    image = Image.open(image_path).convert("RGBA")
    alpha = image.split()[-1]  # 获取 alpha 通道

    # 将 alpha 转换为 numpy 数组
    alpha_np = np.array(alpha)

    # 创建掩码图：大于阈值为白（255），小于为黑（0）
    mask = (alpha_np > alpha_threshold).astype(np.uint8) * 255

    # 保存为灰度图
    mask_image = Image.fromarray(mask, mode='L')
    mask_image.save(output_path)
    print(f"Mask saved: {output_path}")

def batch_convert(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            create_binary_mask(input_path, output_path)

if __name__ == "__main__":
    # 设置你的输入/输出文件夹路径
    input_dir = "/Users/dvd/Desktop/Workspace/U-2-Net/new_train_data/temp/processed_images"
    output_dir = "/Users/dvd/Desktop/Workspace/U-2-Net/new_train_data/temp/masks"

    batch_convert(input_dir, output_dir)