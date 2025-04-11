import os
import requests
from PIL import Image
from io import BytesIO
    
def download_image(url, index, output_folder):
    url = url.strip()
    img_name = os.path.join(output_folder, f'image_{index+1}.png')  # 使用固定的索引命名文件
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # 打开并转换为 RGBA，保留透明通道
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        image.save(img_name, format='PNG')
        print(f'下载图片 {index+1} 成功')

        return True
    except Exception as e:
        # 即使下载失败，也创建一个空文件占位
        with open(img_name, 'w') as f:
            pass
        print(f'下载图片 {index+1} 失败: {e}')
        return False
    
def download_images_from_txt(txt_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(txt_file, 'r', encoding='utf-8') as file:
        urls = [line.strip() for line in file if line.strip()]

    for i, url in enumerate(urls):
        download_image(url, i, output_folder)


if __name__ == '__main__':
    request_urls_path = "../datas/gaoding_urls/request_urls.txt"
    request_images_path = "../datas/gaoding_images/request_images"

    response_urls_path = "../datas/gaoding_urls/response_urls.txt"
    response_images_path = "../datas/gaoding_images/response_images"
    # 执行下载
    download_images_from_txt(request_urls_path, request_images_path)
    download_images_from_txt(response_urls_path, response_images_path)
