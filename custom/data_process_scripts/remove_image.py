import os

# 替换为你的目录路径
folder_a = "../../new_train_data/temp/request_images"
folder_b = "../../new_train_data/temp/response_images"

# 获取 folder_a 中当前还存在的图片文件名
files_in_a = set(os.listdir(folder_a))

# 遍历 folder_b 中的文件
for filename in os.listdir(folder_b):
    file_path_b = os.path.join(folder_b, filename)

    # 如果该文件在 folder_a 中已经不存在，就删除
    if filename not in files_in_a:
        os.remove(file_path_b)
        print(f"已删除：{file_path_b}")