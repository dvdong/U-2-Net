import csv
import json
import re
import tkinter as tk
from tkinter import filedialog
import os

log_path = "../datas/log/gaoding.csv"
output_dir = "../datas/gaoding_urls"

def parse_log(log: str):
    """解析日志，提取 REQUEST 和 RESPONSE 的 URL"""
    print("Parsing log:", log)  # 打印原始日志，检查格式

    try:
        # 解析外层 JSON
        log_data = json.loads(log)
        msg = log_data.get("msg", "")  
    except json.JSONDecodeError as e:
        print("外层 JSON 解析失败:", e)
        return "Not Found", "Not Found"

    print("Extracted msg:", msg)  # 确保 `msg` 格式正确

    # **使用正则表达式提取 REQUEST 和 RESPONSE JSON 片段**
    request_match = re.search(r'\[\*\*\*REQUEST\*\*\*\], ({.*?})[;\s]', msg)
    response_match = re.search(r'\[\*\*\*RESPONSE\*\*\*\], ({.*?})[;\s]', msg)

    if not request_match or not response_match:
        print("未找到 REQUEST 或 RESPONSE")
        return "Not Found", "Not Found"

    request_json = request_match.group(1)
    response_json = response_match.group(1)

    # **处理转义字符**
    request_json = request_json.replace('\\/', '/')
    response_json = response_json.replace('\\/', '/')

    print("Extracted REQUEST JSON:", request_json)
    print("Extracted RESPONSE JSON:", response_json)

    try:
        request_data = json.loads(request_json)
    except json.JSONDecodeError as e:
        print("REQUEST JSON 解析失败:", e)
        request_data = {}

    try:
        response_data = json.loads(response_json)
    except json.JSONDecodeError as e:
        print("RESPONSE JSON 解析失败:", e)
        response_data = {}

    logo_url = request_data.get("logo", "Not Found")
    logo_cutout_url = response_data.get("data", {}).get("logoCutout", "Not Found")

    return logo_url, logo_cutout_url
    
# 把log文件处理为request和response的url
def process_csv(file_path):
    if not file_path:
        print("No file selected.")
      
    request_urls = []
    response_urls = []
    
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            log_text = row.get("log", "")
            if log_text:
                request_url, response_url = parse_log(log_text)
                request_urls.append(request_url)
                response_urls.append(response_url)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "request_urls.txt"), "w", encoding="utf-8") as req_file:
        req_file.write("\n".join(request_urls))
    
    with open(os.path.join(output_dir, "response_urls.txt"), "w", encoding="utf-8") as res_file:
        res_file.write("\n".join(response_urls))
    
    print("Processing complete. Check request_urls.txt and response_urls.txt")


# 运行脚本
if __name__ == "__main__":
    file_path = log_path
    process_csv(file_path=file_path)
