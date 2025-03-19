import torch

# 加载原始权重
original_state_dict = torch.load("./best_model.pth", map_location='cpu')

# 移除包含 'total_ops' 和 'total_params' 的键
cleaned_state_dict = {k: v for k, v in original_state_dict.items() if not k.endswith(('total_ops', 'total_params'))}
torch.save(cleaned_state_dict, "./cleaned_best_model.pth")