import os
import json
from PIL import Image
import torch
import numpy as np
from models.archs.RetinexFormer_arch import RetinexFormer 

def process_and_save_images(input_dir, output_dir, model_path, device):
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    with open('config.json') as f: 
        config = json.load(f) 
    model = RetinexFormer(**config['model']['params']).to(device)
    if os.path.exists(model_path):
        print(f"Loading pretrained weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    
    model.eval() 
    
    # 获取输入文件夹中的所有图片文件
    file_list = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for file_name in file_list:
        # 读取图片
        input_path = os.path.join(input_dir, file_name)
        img = np.array(Image.open(input_path)).astype(np.float32) / 255.0
        
        height, width, _ = img.shape
        
        # 判断是否需要裁切
        if height > 2000 and width > 3000:
            print(f"Image {file_name} is larger than 2000x3000. Splitting into 4 parts.")
            
            # 分割图像为四部分
            half_height, half_width = height // 2, width // 2
            parts = [
                img[:half_height, :half_width],  # 左上
                img[:half_height, half_width:],  # 右上
                img[half_height:, :half_width],  # 左下
                img[half_height:, half_width:]   # 右下
            ]
            
            processed_parts = []
            for part in parts:
                # 转换为 CHW 格式并添加 batch 维度
                part_tensor = torch.from_numpy(part).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # 前向传播
                with torch.no_grad():
                    output_tensor = model(part_tensor)
                
                # 处理输出
                output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 去除 batch 维度并转为 HWC 格式
                output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)  # 将像素值限制在 [0, 255] 并转换为 uint8
                
                processed_parts.append(output_np)
            
            # 重组图像
            top = np.concatenate((processed_parts[0], processed_parts[1]), axis=1)
            bottom = np.concatenate((processed_parts[2], processed_parts[3]), axis=1)
            output_np = np.concatenate((top, bottom), axis=0)
        else:
            # 转换为 CHW 格式并添加 batch 维度
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # 前向传播
            with torch.no_grad():
                output_tensor = model(img_tensor)
            
            # 处理输出
            output_np = output_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # 去除 batch 维度并转为 HWC 格式
            output_np = np.clip(output_np * 255.0, 0, 255).astype(np.uint8)  # 将像素值限制在 [0, 255] 并转换为 uint8
        
        # 保存处理后的图片
        output_img = Image.fromarray(output_np)
        output_path = os.path.join(output_dir, file_name)
        output_img.save(output_path)
        print(f"Processed and saved: {output_path}")
        
        torch.cuda.empty_cache() 

# 示例调用
if __name__ == '__main__':
    input_dir = "/home/alic-li/RetinexRWKV/datasets/test_2025/input/" 
    output_dir = "/home/alic-li/RetinexRWKV/datasets/test_2025/output/"
    model_path = "./best_model_nfeat_16.pth" 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    process_and_save_images(input_dir, output_dir, model_path, device)