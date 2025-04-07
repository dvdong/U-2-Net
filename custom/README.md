# 项目结构

- `custom`  
原项目之外的自定义逻辑, 包括数据处理、脚本，用于训练自己的模型和搭建服务

- `figures`  
项目的一些示例图片
- `gradio`  
使用gradio搭建的一个快速Web Demo, 提供UI界面查看抠图效果
- `model`  
模型的代码
- `saved_models`  
存放模型的地方 
- `test_data`  
测试数据集 内部根据不同任务区分子文件夹

    - `data_loader.py`  
    加载数据  
    
    - `setup_model_weights.py`  
    设置模型权重  

    -  `u2net_human_seg_test.py`  
    人像分割测试  

    - `u2net_portrait_composite.py`  

    - `u2net_portrait_demo.py`  

    - `u2net_portrait_test.py`  

    - `u2net_test.py`   
    u2net模型测试脚本  

    - `u2net_train.py`  
    u2net模型训练脚本  

# 环境配置
最好使用虚拟环境  
使用Python 3.0.10   
安装依赖 `pip install -r requirements.txt`  

如果遇到`pip`下载证书错误, 执行 `pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org`

# 数据处理
1. 处理log文件，解析出原图和抠图后url
2. 分别下载原图和抠图后图片
3. 处理抠图后图片为mask文件
4. 生成训练集