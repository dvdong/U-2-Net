import os
import shutil
import random

# 配置路径
image_dir = './images1/request_images'
mask_dir = './images1/masks'

output_base = 'dataset_split'
train_ratio = 0.9

# 收集所有文件名（假设 mask 和 image 同名）
filenames = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
filenames.sort()  # 可选：排序保持一致性
random.shuffle(filenames)  # 打乱顺序

# 计算数量
total = len(filenames)
train_count = int(train_ratio * total)

train_files = filenames[:train_count]
test_files = filenames[train_count:]

def copy_files(file_list, set_name):
    for f in file_list:
        # 拷贝 image
        os.makedirs(f'{output_base}/{set_name}/images', exist_ok=True)
        shutil.copy(f'{image_dir}/{f}', f'{output_base}/{set_name}/images/{f}')
        # 拷贝 mask
        os.makedirs(f'{output_base}/{set_name}/masks', exist_ok=True)
        shutil.copy(f'{mask_dir}/{f}', f'{output_base}/{set_name}/masks/{f}')

copy_files(train_files, 'train')
copy_files(test_files, 'test')

print(f"划分完成：Train={len(train_files)}，Test={len(test_files)}")