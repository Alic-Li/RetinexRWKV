########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math
import torch
import torch.nn as nn
from torch.nn import functional as F

MyModule = torch.jit.ScriptModule
MyFunction = torch.jit.script_method

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = 4
CHUNK_LEN = 16
flags = [f'-D_C_={HEAD_SIZE}', f"-D_CHUNK_LEN_={CHUNK_LEN}",'-xhip', '-fopenmp', '-ffast-math', '-O3', '--offload-arch=gfx1100','-munsafe-fp-atomics',]
load(name="wind_backstepping", sources=[f'models/archs/cuda/wkv7_cuda.cu', 'models/archs/cuda/wkv7_op.cu'], is_python_module=False, verbose=True, extra_cuda_cflags=flags)
class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w,q,k,v,z,b):
        # print(w.dtype, q.dtype, k.dtype, v.dtype, z.dtype, b.dtype)
        B,T,H,C = w.shape 
        assert T%CHUNK_LEN == 0 # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
        assert all(i.dtype==torch.bfloat16 for i in [w,q,k,v,z,b])
        assert all(i.is_contiguous() for i in [w,q,k,v,z,b])
        y = torch.empty_like(v)
        s = torch.empty(B,H,T//CHUNK_LEN,C,C, dtype=torch.float32,device=w.device)
        sa = torch.empty(B,T,H,C, dtype=torch.float32,device=w.device)
        torch.ops.wind_backstepping.forward(w,q,k,v,z,b, y,s,sa)
        ctx.save_for_backward(w,q,k,v,z,b,s,sa)
        return y
    @staticmethod
    def backward(ctx, dy):
        assert all(i.dtype==torch.bfloat16 for i in [dy])
        assert all(i.is_contiguous() for i in [dy])
        w,q,k,v,z,b,s,sa = ctx.saved_tensors
        dw,dq,dk,dv,dz,db = [torch.empty_like(x) for x in [w,q,k,v,z,b]]
        torch.ops.wind_backstepping.backward(w,q,k,v,z,b, dy,s,sa, dw,dq,dk,dv,dz,db)
        return dw,dq,dk,dv,dz,db
def RUN_CUDA_RWKV7g(q,w,k,v,a,b):
    B,T,HC = q.shape
    q,w,k,v,a,b = [i.view(B,T,HC//4,4) for i in [q,w,k,v,a,b]]
    return WindBackstepping.apply(w,q,k,v,a,b).view(B,T,HC)
########################################################################################################

class RWKV_Tmix_x070(MyModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(torch.ones(1, 1, C))
            self.x_w = nn.Parameter(torch.ones(1, 1, C))
            self.x_k = nn.Parameter(torch.ones(1, 1, C))
            self.x_v = nn.Parameter(torch.ones(1, 1, C))
            self.x_a = nn.Parameter(torch.ones(1, 1, C))
            self.x_g = nn.Parameter(torch.ones(1, 1, C))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                    nn.init.orthogonal_(x, gain=gain * scale)
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C-1) - 0.5
                zigzag[n] = ((n % N) - ((N-1) / 2)) / ((N-1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])

            D_DECAY_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1,1,C) + 0.5 + zigzag*2.5) # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round(  (2.5*(C**0.5))  /32)*32)) # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1,1,C)-0.19 + zigzag*0.3 + linear*0.4)

            D_MV_LORA = max(32, int(round(  (1.7*(C**0.5))  /32)*32)) # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1,1,C)+0.73 - linear*0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round(  (5*(C**0.5))  /32)*32)) # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1,1,C)+0.71 - linear*0.1)
            self.k_a = nn.Parameter(torch.zeros(1,1,C)+1.02)
            self.r_k = nn.Parameter(torch.zeros(H,N)-0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5) # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.key.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.value.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.output.weight.data.zero_()  # 如果测试发现输出tensor全是0，则将此处注释就不为0了，训练时建议初始化为0

    # @MyFunction
    def forward(self, x, illu_fea):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5 # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        v = v * illu_fea
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2) # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B,T,H,-1), dim=-1, p=2.0).view(B,T,C)
        k = k * (1 + (a-1) * self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk*a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + ((r.view(B,T,H,-1)*k.view(B,T,H,-1)*self.r_k).sum(dim=-1, keepdim=True) * v.view(B,T,H,-1)).view(B,T,C)
        x = self.output(x * g)
        return x
    

if __name__ == "__main__":    
    import argparse
    args = argparse.Namespace(
        head_size=16,      # 每个头的大小
        dim_att=64,       # 总的注意力维度 
        n_embd=64         # 嵌入维度
    )
    print(f"配置: {args.dim_att // args.head_size} 个头，每个头 {args.head_size} 维")
    print(f"总嵌入维度: {args.n_embd}")
    
    device = torch.device("cuda")
    print(f"使用设备: {device}")
    
    model = RWKV_Tmix_x070(args).to(device)
    
    batch_size = 2
    seq_length = 65536  # 必须是16的倍数
    feature_dim = args.n_embd
    
    input_tensor = torch.randn(batch_size, seq_length, feature_dim, 
                              dtype=torch.bfloat16, device=device).contiguous()
    illu_fea = torch.randn(batch_size, seq_length, feature_dim, 
                          dtype=torch.bfloat16, device=device).contiguous()
    
    print(f"\n输入张量形状: {input_tensor.shape}")
    print(f"输入张量数据类型: {input_tensor.dtype}")
    print(f"输入张量设备: {input_tensor.device}")
    print(f"输入张量是否连续: {input_tensor.is_contiguous()}")
    
    model = model.bfloat16()
    
    with torch.no_grad():
        output = model(input_tensor, illu_fea)
    
    print(f"输出张量形状: {output.shape}")
    print(f"输出张量数据类型: {output.dtype}")
    print(f"输出张量设备: {output.device}")
    
    print("\n测试不同序列长度:")
    for test_seq_len in [16, 32, 64]:
        if test_seq_len % 16 != 0:
            print(f"  序列长度 {test_seq_len}: 跳过 (必须是16的倍数)")
            continue
            
        test_input = torch.randn(1, test_seq_len, args.n_embd, 
                                dtype=torch.bfloat16, device=device).contiguous()
        illu_fea = torch.randn(1, test_seq_len, args.n_embd, 
                                dtype=torch.bfloat16, device=device).contiguous()
        with torch.no_grad():
            test_output = model(test_input, illu_fea)
        print(f"  序列长度 {test_seq_len}: 输入{test_input.shape} -> 输出{test_output.shape}")
    print(output)
