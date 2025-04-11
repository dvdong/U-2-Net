from PIL import Image
import os

def apply_alpha_mask(original_dir: str, mask_dir: str, output_dir: str):
    """
    为original_dir中的图像应用mask_dir中的灰度图作为Alpha通道，
    生成带透明背景的图像并保存在output_dir中。
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取原图和分割图文件名列表
    image_filenames = sorted(os.listdir(original_dir))
    mask_filenames = sorted(os.listdir(mask_dir))

    for img_name, mask_name in zip(image_filenames, mask_filenames):
        img_path = os.path.join(original_dir, img_name)
        mask_path = os.path.join(mask_dir, mask_name)
        output_path = os.path.join(output_dir, img_name)

        # 打开原图和分割图
        image = Image.open(img_path).convert("RGBA")  # 转为RGBA，确保有Alpha通道
        mask = Image.open(mask_path).convert("L")      # 转为灰度图

        # 归一化mask（确保是0-255范围）
        mask = mask.point(lambda p: p * (255.0 / mask.getextrema()[1]))

        # 组合原图和Alpha通道
        r, g, b, _ = image.split()
        result = Image.merge("RGBA", (r, g, b, mask))

        # 保存抠好的图
        result.save(output_path, format="PNG")
        print(f"Processed: {output_path}")

    print("All images processed!")

# 如果作为脚本运行
if __name__ == "__main__":
    # 原图和分割图的目录
    original_dir = "../dataset_split/test/images"
    mask_dir = "../dataset_split/u2net_results"
    output_dir = "../dataset_split/results_0.48"

    apply_alpha_mask(original_dir, mask_dir, output_dir)