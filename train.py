import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomCrop
from models.archs.RetinexFormer_arch import RetinexFormer
from models.lr_scheduler import CosineAnnealingRestartLR
from skimage.metrics import structural_similarity as ssim
import os
from PIL import Image
from tqdm import tqdm 
from thop import profile


class PairedDataset(Dataset):
    def __init__(self, input_dir, gt_dir, patch_size=256, train=True):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.train = train
        
        self.file_list = [f for f in os.listdir(input_dir) 
                         if f.endswith(('.png'))]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.file_list[idx])
        gt_path = os.path.join(self.gt_dir, self.file_list[idx])
        
        input_img = np.array(Image.open(input_path)).astype(np.float32) / 255.0
        gt_img = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        
        if self.train:
            # 随机裁剪
            h, w, _ = input_img.shape
            i = np.random.randint(0, h - self.patch_size)
            j = np.random.randint(0, w - self.patch_size)
            
            input_img = input_img[i:i+self.patch_size, j:j+self.patch_size]
            gt_img = gt_img[i:i+self.patch_size, j:j+self.patch_size]
        
        # HWC to CHW
        input_tensor = torch.from_numpy(input_img).permute(2,0,1)
        gt_tensor = torch.from_numpy(gt_img).permute(2,0,1)
        
        return input_tensor, gt_tensor

def validate(model, val_loader, device):
    model.eval()
    psnr_list = []
    ssim_list = []
    
    # 使用 tqdm 包装验证数据加载器
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # 计算指标
            outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
            targets_np = targets.numpy().transpose(0, 2, 3, 1)
            
            for out, gt in zip(outputs_np, targets_np):
                # 去除第一个维度
                out = out[0] if out.shape[0] == 1 else out
                gt = gt[0] if gt.shape[0] == 1 else gt

                # print(out.shape, gt.shape)                
                # 计算 PSNR
                psnr = 10 * np.log10(1.0 / np.mean((out - gt)**2))
                psnr_list.append(psnr)
                
                # 计算 SSIM
                ssim_value = ssim(out, gt, channel_axis = 2, multichannel=True, data_range=1.0)
                ssim_list.append(ssim_value)
                  
    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    # 读取配置
    with open('config.json') as f: 
        config = json.load(f)  
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 解析数据集路径
    train_input_dir = os.path.join(config['train']['dataset']['train_dir'], 'Input')
    train_gt_dir = os.path.join(config['train']['dataset']['train_dir'], 'GT')
    val_input_dir = os.path.join(config['train']['dataset']['val_dir'], 'Input')
    val_gt_dir = os.path.join(config['train']['dataset']['val_dir'], 'GT')
    
    # 初始化数据加载器
    train_dataset = PairedDataset(train_input_dir, train_gt_dir,
                                  patch_size=config['train']['patch_size'],
                                  train=True)
    val_dataset = PairedDataset(val_input_dir, val_gt_dir,
                                patch_size=None, train=False)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=config['train']['batch_size'],
                              shuffle=True,
                              num_workers=config['train']['num_workers'])
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=1)
    
    # 创建模型
    model = RetinexFormer(**config['model']['params']).to(device)
    pretrained_weights_path = config['train'].get('pretrained_weights', None)
    if pretrained_weights_path and os.path.exists(pretrained_weights_path):
        print(f"Loading pretrained weights from {pretrained_weights_path}")
        model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
    else:
        print("No pretrained weights found or specified.")
    print("Training Start!!!")
    # inputs = torch.randn((1, 3, 256, 256)).cuda()
    # flops, params = profile(model, inputs=(inputs,))
    # print(f"Total FLOPs: {flops / 1e9:.2f} GFLOPs")  # 转换为 GFLOPs
    # print(f"Total Parameters: {params / 1e6:.2f} M")  # 转换为百万参数量
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=config['train']['lr'],
                                 betas=config['optimizer']['betas'])
    
    # 学习率调度器
    scheduler = CosineAnnealingRestartLR(optimizer,
                                         periods=config['scheduler']['periods'],
                                         restart_weights=config['scheduler']['restart_weights'],
                                         eta_min=config['scheduler']['eta_min'])
    
    best_psnr = 0
    best_ssim = 0
    for epoch in range(config['train']['epochs']):
        model.train()
        total_loss_epoch = 0  # 用于记录每个 epoch 的总 loss
        num_batches = len(train_loader)
        
        # 使用 tqdm 包装训练数据加载器
        train_loop = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {epoch+1}/{config['train']['epochs']}")
        for batch_idx, (inputs, targets) in train_loop:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            l1_loss = torch.nn.L1Loss()(outputs, targets)
            
            # 反向传播
            optimizer.zero_grad()
            l1_loss.backward()
            optimizer.step()
            
            # 更新进度条描述信息
            total_loss_epoch += l1_loss.item()
            avg_loss = total_loss_epoch / (batch_idx + 1)
            train_loop.set_postfix(loss=f"{avg_loss:.4f}")
        if (epoch + 1) % config["train"]["val_epoch"] == 0:
            # 验证
            val_psnr, val_ssim = validate(model, val_loader, device)

            # 保存最佳模型
            if val_ssim > best_ssim:
                best_ssim = val_ssim
                torch.save(model.state_dict(), 'best_model.pth')
            print(f'Epoch [{epoch+1}/{config["train"]["epochs"]}] '
                  f'Val PSNR: {val_psnr:.2f} Val SSIM: {val_ssim:.4f} Best SSIM: {best_ssim:.4f}')
        
        # 更新学习率
        scheduler.step()


if __name__ == '__main__':
    main()