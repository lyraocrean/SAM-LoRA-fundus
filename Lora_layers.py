#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
#  7.25 : 从 convlora 那里直接复制来的
#         只留了 LoRAlayer 和 ConvLoRA 类, 其他全部注释掉了
#  7.30 : 选择使用LinearLoRA
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import sys  # 确保添加这行
import math
from typing import Optional, List
from IPython import embed
import types
import importlib

class LoRALayer(nn.Module):
    def __init__(self, sam_model, r=4, lora_alpha=4, lora_dropout=0.1):
        # 添加参数验证
        if not hasattr(sam_model, 'image_encoder'):
            raise TypeError("sam_model必须包含image_encoder属性")
        super().__init__()
        self.sam_model = sam_model
        self.image_encoder = sam_model.image_encoder
        self.prompt_encoder = sam_model.prompt_encoder
        self.mask_decoder = sam_model.mask_decoder
        
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        # 初始化 LoRA 参数
        self._init_lora_parameters()
        
        print(f"接收到的sam_model类型: {type(sam_model)}")
        print(f"sam_model包含的属性: {dir(sam_model)}")
    
    def _init_lora_parameters(self):
        """初始化 LoRA 相关参数，适配SAM模型结构"""
        # 冻结所有SAM模型参数
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        # 为图像编码器的每个注意力块添加LoRA层
        blocks = self.image_encoder.blocks
        self.lora_A_qkv = nn.ModuleList()
        self.lora_B_qkv = nn.ModuleList()
        
        # 对每个transformer块添加LoRA参数 - 适配SAM的QKV一体化结构
        for i, block in enumerate(blocks):
            # 检查注意力机制的实现
            if hasattr(block.attn, 'qkv'):
                # 如果使用合并的qkv投影
                qkv_weight_shape = block.attn.qkv.weight.shape
                in_dim = qkv_weight_shape[1]
                out_dim = qkv_weight_shape[0]
                
                # 创建LoRA参数用于qkv投影
                lora_A = nn.Linear(in_dim, self.r, bias=False)
                lora_B = nn.Linear(self.r, out_dim, bias=False)
                
                # 使用较小的初始化值
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)
                
                self.lora_A_qkv.append(lora_A)
                self.lora_B_qkv.append(lora_B)
            elif hasattr(block.attn, 'proj'):
                # 如果注意力机制使用分离的q,k,v和一个输出投影
                proj_weight_shape = block.attn.proj.weight.shape
                in_dim = proj_weight_shape[1]
                out_dim = proj_weight_shape[0]
                
                # 创建LoRA参数用于输出投影
                lora_A = nn.Linear(in_dim, self.r, bias=False)
                lora_B = nn.Linear(self.r, out_dim, bias=False)
                
                nn.init.kaiming_uniform_(lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(lora_B.weight)
                
                self.lora_A_qkv.append(lora_A)
                self.lora_B_qkv.append(lora_B)
            else:
                raise ValueError(f"无法识别的注意力机制结构: {block.attn}")
        
        # 替换原始前向传播方法
        self._replace_forward_methods()
    
    def _replace_forward_methods(self):
        """替换原始模块的前向传播方法以添加LoRA路径"""
        # 创建闭包引用
        lora_A_qkv = self.lora_A_qkv
        lora_B_qkv = self.lora_B_qkv
        lora_alpha = self.lora_alpha
        r = self.r
        lora_dropout = self.lora_dropout
        
        # 保存对类实例的引用，以便在闭包中使用
        lora_self = self
        
        # 定义新的注意力前向传播方法，适配SAM模型的注意力机制
        def new_attn_forward(self_attn, x):
            # 查找当前注意力块在模型中的索引
            block_idx = -1
            for i, block in enumerate(lora_self.image_encoder.blocks):
                if block.attn is self_attn:
                    block_idx = i
                    break
            
            if block_idx == -1:
                raise ValueError("无法确定当前注意力块的索引")
            
            # 处理SAM的注意力机制 - 假设SAM使用一个proj层作为输出投影
            if hasattr(self_attn, 'proj'):
                # 保存原始实现逻辑
                # 注意：这里保留原始前向传播的大部分逻辑
                # 仅修改关键部分以添加LoRA路径
                B, N, C = x.shape
                
                # 让原始QKV计算执行
                qkv = self_attn.qkv(x)
                qkv = qkv.reshape(B, N, 3, self_attn.num_heads, C // self_attn.num_heads)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                # 计算注意力
                attn = (q @ k.transpose(-2, -1)) * self_attn.scale
                attn = attn.softmax(dim=-1)
                attn = self_attn.attn_drop(attn)
                
                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                
                # 应用原始投影
                orig_output = self_attn.proj(x)
                
                # 应用LoRA路径到投影输出
                lora_output = lora_B_qkv[block_idx](lora_dropout(lora_A_qkv[block_idx](x))) * (lora_alpha / r)
                
                # 合并原始输出和LoRA输出
                output = orig_output + lora_output
                output = self_attn.proj_drop(output)
                
                return output
            else:
                # 如果结构与预期不符，则使用原始方法
                print(f"警告：未能为块 {block_idx} 添加LoRA路径，使用原始实现")
                return self_attn._original_forward(x)
        
        # 为每个注意力块保存原始前向方法并应用新方法
        for i, block in enumerate(self.image_encoder.blocks):
            # 保存原始方法
            if not hasattr(block.attn, '_original_forward'):
                block.attn._original_forward = block.attn.forward
            
            # 应用新方法
            block.attn.forward = types.MethodType(new_attn_forward, block.attn)
        
        # 确保所有LoRA参数都是可训练的
        for param in self.lora_A_qkv.parameters():
            param.requires_grad = True
        for param in self.lora_B_qkv.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # 使用修改后的SAM模型进行前向传播
        return self.sam_model(x)

    def parameters(self):
        # 返回需要训练的参数
        return self.sam_model.parameters()

class ConvLoRA(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        kernel_size: int,
        r: int = 4, 
        lora_alpha: int = 4, 
        lora_dropout: float = 0.1,
        stride: int = 1,
        padding: int = 0,
        **kwargs
    ):
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        
        self.lora_A = nn.Parameter(
            torch.zeros((r, in_channels * kernel_size))
        )
        self.lora_B = nn.Parameter(
            torch.zeros((out_channels, r))
        )
        self.scaling = self.lora_alpha / self.r
        
        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor):
        batch_size, in_channels, length = x.shape
        
        # 展开输入用于 LoRA 计算
        x_unfolded = F.unfold(
            x.unsqueeze(-1),
            (self.kernel_size, 1),
            stride=(self.stride, 1),
            padding=(self.padding, 0)
        )
        
        # LoRA path
        out = (
            self.lora_dropout(x_unfolded.transpose(1, 2))
            @ self.lora_A.T
            @ self.lora_B.T
            * self.scaling
        )
        
        return out.transpose(1, 2).view(batch_size, self.out_channels, -1)

class Linear(nn.Linear):
    """使用组合模式而非多重继承实现LoRA功能的线性层"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.1,
        merge_weights: bool = True,
        fan_in_fan_out: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        
        # 初始化LoRA相关参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.merge_weights = merge_weights
        self.fan_in_fan_out = fan_in_fan_out
        # 添加初始化merged属性
        self.merged = False
        
        if r > 0:
            # 修改为使用ParameterList来存储LoRA参数
            # 这样可以通过索引访问，并与add_lora.py中的代码兼容
            # Q, K, V 各有一个lora参数
            self.lora_A = nn.ParameterList([
                nn.Parameter(torch.zeros((r, in_features))),  # Q
                nn.Parameter(torch.zeros((r, in_features))),  # K
                nn.Parameter(torch.zeros((r, in_features)))   # V
            ])
            
            # 计算每个部分的输出维度
            third_dim = out_features // 3
            self.lora_B = nn.ParameterList([
                nn.Parameter(torch.zeros((third_dim, r))),  # Q
                nn.Parameter(torch.zeros((third_dim, r))),  # K
                nn.Parameter(torch.zeros((third_dim, r)))   # V
            ])
            
            self.scaling = self.lora_alpha / self.r
            
            # 初始化LoRA权重
            for a_param in self.lora_A:
                nn.init.kaiming_uniform_(a_param, a=math.sqrt(5))
            for b_param in self.lora_B:
                nn.init.zeros_(b_param)
            
            if lora_dropout > 0:
                self.lora_dropout = nn.Dropout(p=lora_dropout)
            else:
                self.lora_dropout = nn.Identity()
        
        # 冻结原始权重
        self.weight.requires_grad = False
    
    # 添加train方法来正确处理权重合并
    def train(self, mode=True):
        """处理训练模式切换时的权重合并/分离逻辑"""
        super().train(mode)
        if mode:
            # 训练模式: 如果权重已合并，需要分离
            if self.merge_weights and self.merged:
                # 分离权重
                if self.r > 0:
                    self.weight.data -= self.get_lora_weight_delta()
                self.merged = False
        else:
            # 评估模式: 如果需要合并且尚未合并，则合并权重
            if self.merge_weights and not self.merged:
                # 合并权重
                if self.r > 0:
                    self.weight.data += self.get_lora_weight_delta()
                self.merged = True
        return self
    
    def get_lora_weight_delta(self):
        """计算LoRA权重变化量"""
        if self.r > 0:
            # 创建一个空张量来存储合并的权重变化
            delta_w = torch.zeros_like(self.weight)
            
            # 计算每个部分(Q,K,V)的权重变化并填充到相应位置
            third_dim = self.weight.shape[0] // 3
            for idx, (lora_a, lora_b) in enumerate(zip(self.lora_A, self.lora_B)):
                # 计算这部分的权重变化
                part_delta = (lora_b @ lora_a) * self.scaling
                # 填充到相应位置
                start_idx = idx * third_dim
                end_idx = (idx + 1) * third_dim
                delta_w[start_idx:end_idx, :] = part_delta
                
            return delta_w
        return 0
    
    def forward(self, x):
        if self.r > 0 and not self.merged:
            # 应用原始线性层
            result = F.linear(x, self.weight, bias=self.bias)
            
            # 应用LoRA路径 - 分别处理Q,K,V
            if self.r > 0:
                # 创建一个与result相同形状的零张量
                lora_output = torch.zeros_like(result)
                
                # 处理Q,K,V每个部分
                third_dim = self.weight.shape[0] // 3
                for idx, (lora_a, lora_b) in enumerate(zip(self.lora_A, self.lora_B)):
                    # 应用dropout到输入
                    dropped_x = self.lora_dropout(x)
                    
                    # 计算这部分的LoRA输出
                    part_output = dropped_x @ lora_a.t() @ lora_b.t() * self.scaling
                    
                    # 确保part_output的维度正确
                    if part_output.shape[-1] != third_dim:
                        part_output = part_output.reshape(*part_output.shape[:-1], third_dim)
                    
                    # 填充到相应位置
                    start_idx = idx * third_dim
                    end_idx = (idx + 1) * third_dim
                    lora_output[..., start_idx:end_idx] = part_output
                
                # 合并原始输出和LoRA输出
                result = result + lora_output
            
            return result
        else:
            # 正常前向传播(如果r=0或权重已合并)
            return F.linear(x, self.weight, bias=self.bias)

# 现在可以定义 Conv1d，因为 ConvLoRA 已经定义
class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# class Embedding(nn.Embedding, LoRALayer):
#     # LoRA implemented in a dense layer
#     def __init__(
#         self,
#         num_embeddings: int,
#         embedding_dim: int,
#         r: int = 0,
#         lora_alpha: int = 1,
#         merge_weights: bool = True,
#         **kwargs
#     ):
#         nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
#                            merge_weights=merge_weights)
#         # Actual trainable parameters
#         if r > 0:
#             self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
#             self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
#             self.scaling = self.lora_alpha / self.r
#             # Freezing the pre-trained weight matrix
#             self.weight.requires_grad = False
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         nn.Embedding.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             # initialize A the same way as the default for nn.Linear and B to zero
#             nn.init.zeros_(self.lora_A)
#             nn.init.normal_(self.lora_B)
#
#     def train(self, mode: bool = True):
#         nn.Embedding.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 # Make sure that the weights are not merged
#                 if self.r > 0:
#                     self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 # Merge the weights and mark it
#                 if self.r > 0:
#                     self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
#                 self.merged = True
#         
#         def forward(self, x: torch.Tensor):
#             if self.r > 0 and not self.merged:
#                 result = nn.Embedding.forward(self, x)
#                 after_A = F.embedding(
#                     x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
#                     self.norm_type, self.scale_grad_by_freq, self.sparse
#                 )
#                 result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
#                 return result
#             else:
#                 return nn.Embedding.forward(self, x)           

# class MergedLinear(nn.Linear, LoRALayer):
#     def __init__(self, 
#                  in_features: int, 
#                  out_features: int, 
#                  r: int = 0, 
#                  lora_alpha: int = 1, 
#                  lora_dropout: float = 0.,
#                  enable_lora: List[bool] = [False, False, False],  # [Q, K, V]
#                  fan_in_fan_out: bool = False,
#                  merge_weights: bool = True,
#                  **kwargs):
#         nn.Linear.__init__(self, in_features, out_features, **kwargs)
#         LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
#                            merge_weights=merge_weights)
#         
#         assert len(enable_lora) == 3, "enable_lora必须包含3个布尔值对应Q/K/V分支"
#         self.enable_lora = enable_lora
#         self.fan_in_fan_out = fan_in_fan_out
#         
#         if r > 0 and any(enable_lora):
#             # 为每个启用的分支创建独立参数
#             self.lora_A = nn.ParameterList()
#             self.lora_B = nn.ParameterList()
#             for i, enabled in enumerate(self.enable_lora):
#                 if enabled:
#                     self.lora_A.append(nn.Parameter(torch.zeros(r, in_features)))
#                     self.lora_B.append(nn.Parameter(torch.zeros(out_features//3, r)))
#                 else:
#                     self.lora_A.append(None)
#                     self.lora_B.append(None)
#             self.scaling = self.lora_alpha / r
#             self.weight.requires_grad = False
#             
#         self.reset_parameters()
#         if fan_in_fan_out:
#             self.weight.data = self.weight.data.transpose(0, 1)
#
#     def reset_parameters(self):
#         nn.Linear.reset_parameters(self)
#         if hasattr(self, 'lora_A'):
#             for i, (a, b) in enumerate(zip(self.lora_A, self.lora_B)):
#                 if a is not None and b is not None:
#                     nn.init.kaiming_uniform_(a, a=math.sqrt(5))
#                     nn.init.zeros_(b)
#
#     def merge_AB(self):
#         delta_ws = []
#         for a, b in zip(self.lora_A, self.lora_B):
#             if a is not None and b is not None:
#                 delta_ws.append(b @ a * self.scaling)
#             else:
#                 delta_ws.append(torch.zeros(self.out_features//3, self.in_features))
#         return torch.cat(delta_ws, dim=0)
#
#     def train(self, mode: bool = True):
#         nn.Linear.train(self, mode)
#         if mode:
#             if self.merge_weights and self.merged:
#                 if any(self.enable_lora):
#                     self.weight.data -= self.merge_AB().T
#                 self.merged = False
#         else:
#             if self.merge_weights and not self.merged:
#                 if any(self.enable_lora):
#                     self.weight.data += self.merge_AB().T
#                 self.merged = True
#
#     def forward(self, x: torch.Tensor):
#         def T(w):
#             return w.transpose(0, 1) if self.fan_in_fan_out else w
#         
#         if self.merged:
#             return F.linear(x, T(self.weight), bias=self.bias)
#             
#         result = F.linear(x, T(self.weight), bias=self.bias)
#         
#         if hasattr(self, 'lora_A'):
#             lora_outputs = []
#             for i, (a, b) in enumerate(zip(self.lora_A, self.lora_B)):
#                 if a is not None and b is not None and (not hasattr(self, 'enable_lora') or self.enable_lora[i]):
#                     lora_out = (self.lora_dropout(x) @ a.T @ b.T) * self.scaling
#                     # 处理形状以匹配QKV结构
#                     start_idx = i * (result.shape[-1] // 3)
#                     end_idx = (i + 1) * (result.shape[-1] // 3)
#                     # 创建零张量
#                     combined = torch.zeros_like(result)
#                     # 仅将lora输出添加到对应的位置
#                     combined[..., start_idx:end_idx] = lora_out
#                     lora_outputs.append(combined)
#             
#             # 合并所有lora输出
#             if lora_outputs:
#                 result += sum(lora_outputs)
#         
#         return result

# class Conv2d(ConvLoRA):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

# 更健壮的重载检查
try:
    # 仅在交互式环境中执行
    if 'ipykernel' in sys.modules:
        import sys
        current_module = sys.modules[__name__]
        importlib.reload(current_module)
        importlib.invalidate_caches()
except Exception:
    pass
