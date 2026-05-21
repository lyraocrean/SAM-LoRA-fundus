#  ============================================================ #
#  add_lora.py
#  - 为ViT图像编码器（QKV投影）配备轻量级LoRA适配器以包装SAM。
#  - 只有LoRA参数保持可训练状态；默认情况下，基础SAM权重被冻结。
#  - 可选：如果需要额外的微调，可以解冻mask_decoder（或其他部分）。
#  - 此处的线性LoRA将Q/K/V平均拆分；参数存储在ParameterList中以确保兼容性。
#  - 设计用于接入train_lora_sam.py：实例化LoRA_Sam（sam，r，alpha，dropout）。
#  ============================================================ #

from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry
from Lora_layers import ConvLoRA, Linear

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam
from torch.cuda.amp import autocast, GradScaler
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from segment_anything import Sam



'''
# 原本文件里的lora
class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv
'''
# 定义一个继承自PyTorch nn.Module 的类
class LoRA_Sam(nn.Module):
    """低秩适应(LoRA)版本的SAM模型"""
    def __init__(self, sam_model, r, lora_alpha=1.0, lora_dropout=0.1):
        """初始化LoRA SAM模型
        
        参数:
            sam_model: 原始SAM模型
            r: LoRA的秩
            lora_alpha: LoRA的alpha值
            lora_dropout: LoRA层的dropout率
        """
        super().__init__()
        # 保存原始SAM模型
        self.sam = sam_model
        
        # 初始化LoRA参数
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # 获取图像编码器和遮罩解码器
        self.image_encoder = self.sam.image_encoder
        self.mask_decoder = self.sam.mask_decoder
        
        # 保存LoRA权重
        self.w_As = []  # 用于保存所有LoRA A矩阵
        self.w_Bs = []  # 用于保存所有LoRA B矩阵
        
        self.replaced_modules = {}  # 保存替换的模块
        self.modified = False  # 是否已修改模型

        # 添加调试信息
        print(f"创建LoRA SAM模型: r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        # 应用LoRA到模型
        self.add_lora_to_vision_encoder()

        for name, p in self.sam.named_parameters():
            if "lora_" not in name:
                p.requires_grad = False
 
        
        # 报告可训练参数
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"可训练参数总数: {trainable_params:,}")
        lora_params = sum(p.numel() for module_name, module in self.named_modules() 
                          if isinstance(module, Linear) 
                          for p in module.parameters() if p.requires_grad)
        print(f"LoRA参数总数: {lora_params:,}")

    def add_lora_to_vision_encoder(self):
        """将LoRA应用到图像编码器的部分"""
        
        # 修改自注意力模块的QKV投影层
        for name, module in self.image_encoder.named_modules():
            # 处理Linear类型的qkv投影层
            if name.endswith('qkv') and isinstance(module, nn.Linear):
                # 获取原始层的参数
                in_features = module.in_features
                out_features = module.out_features
                
                # 创建新的带有LoRA的Linear层
                new_lora_layer = Linear(
                    in_features, 
                    out_features, 
                    r=self.r, 
                    lora_alpha=self.lora_alpha, 
                    lora_dropout=self.lora_dropout
                )
                
                # print(f"替换层 {name} - 输入: {in_features}, 输出: {out_features}")
                # print(f"LoRA A形状: {[param.shape for param in new_lora_layer.lora_A]}")
                # print(f"LoRA B形状: {[param.shape for param in new_lora_layer.lora_B]}")
                
                # 设置权重和偏置项
                new_lora_layer.weight.data = module.weight.data
                if module.bias is not None:
                    new_lora_layer.bias.data = module.bias.data
                
                # 保存替换的层
                parent_name = '.'.join(name.split('.')[:-1])
                parent = self.image_encoder.get_submodule(parent_name)
                target_name = name.split('.')[-1]
                self.replaced_modules[name] = module
                
                # 替换原始层
                setattr(parent, target_name, new_lora_layer)
                
                # 保存LoRA参数引用
                for i in range(len(new_lora_layer.lora_A)):
                    self.w_As.append(new_lora_layer.lora_A[i])
                    self.w_Bs.append(new_lora_layer.lora_B[i])
                
                print(f"已将LoRA应用到 {name}, w_As长度: {len(self.w_As)}, w_Bs长度: {len(self.w_Bs)}")
        
        # 设置已修改标志
        self.modified = True

    def reset_parameters(self):
        """初始化LoRA参数"""
        # 添加调试信息
        print(f"重置LoRA参数: w_As长度={len(self.w_As)}, w_Bs长度={len(self.w_Bs)}")
        
        # 对保存的权重进行初始化
        for i, (w_A, w_B) in enumerate(zip(self.w_As, self.w_Bs)):
            # print(f"初始化参数 {i}: 类型 w_A={type(w_A)}, w_B={type(w_B)}")
            if isinstance(w_A, nn.Parameter) and w_A.dim() >= 2:
                # 使用较小的初始化值初始化A
                nn.init.kaiming_uniform_(w_A, a=math.sqrt(5))
                # print(f"已初始化 w_A[{i}], 形状={w_A.shape}")
            else:
                print(f"警告: 跳过 w_A[{i}], 类型={type(w_A)}, 维度={w_A.dim() if isinstance(w_A, torch.Tensor) else 'N/A'}")
            
            if isinstance(w_B, nn.Parameter) and w_B.dim() >= 2:
                # 初始化B为零
                nn.init.zeros_(w_B)
                print(f"已初始化 w_B[{i}], 形状={w_B.shape}")
            else:
                print(f"警告: 跳过 w_B[{i}], 类型={type(w_B)}, 维度={w_B.dim() if isinstance(w_B, torch.Tensor) else 'N/A'}")

    def forward(self, images, points=None, boxes=None, masks=None, multimask_output=True, image_size=1024):
        """模型前向传播"""
        # 输入验证
        if images is None:
            raise ValueError("输入图像不能为None")
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"输入图像必须是torch.Tensor，而不是{type(images)}")
        if images.dim() != 4:
            raise ValueError(f"输入图像必须是4维形状[B,C,H,W]，而不是{images.shape}")
        
        # 确保图像编码器处于训练状态
        self.image_encoder.train()
        # 这个循环把全部 ViT 权重都解冻，等于全参训练。
        # for param in self.image_encoder.parameters():
        #     param.requires_grad = True
        
        # 调整图像大小以匹配期望的输入尺寸
        original_size = images.shape[-2:]
        
        # 处理位置嵌入的调整
        # SAM模型的标准大小为1024
        patch_size = 16  # SAM使用16x16的patch size
        
        # 如果图像尺寸不是1024，需要调整pos_embed
        if original_size[0] != 1024 or original_size[1] != 1024:
            # 创建临时图像编码器供使用
            temp_encoder = self.image_encoder
            
            # 获取原始pos_embed
            if hasattr(temp_encoder, 'pos_embed') and temp_encoder.pos_embed is not None:
                old_pos_embed = temp_encoder.pos_embed
                
                # 计算新的位置嵌入网格大小
                new_grid_h = image_size // patch_size
                new_grid_w = image_size // patch_size
                
                # 保存原始位置嵌入的形状
                original_pos_embed = old_pos_embed.clone()
                
                # 调整位置嵌入大小
                old_grid_size = old_pos_embed.shape[1:3]  # 原始网格大小
                
                # 如果需要调整位置嵌入
                if old_grid_size[0] != new_grid_h or old_grid_size[1] != new_grid_w:
                    print(f"调整位置嵌入从 {old_grid_size} 到 {(new_grid_h, new_grid_w)}")
                    # 对位置嵌入进行插值
                    pos_embed = old_pos_embed.permute(0, 3, 1, 2)  # [1, H, W, C] -> [1, C, H, W]
                    pos_embed = F.interpolate(
                        pos_embed, 
                        size=(new_grid_h, new_grid_w), 
                        mode='bicubic', 
                        align_corners=False
                    )
                    pos_embed = pos_embed.permute(0, 2, 3, 1)  # [1, C, H, W] -> [1, H, W, C]
                    
                    # 临时替换位置嵌入
                    temp_encoder.pos_embed = nn.Parameter(pos_embed)
                    
                    # 在图像处理后恢复原始位置嵌入
                    def restore_pos_embed():
                        temp_encoder.pos_embed = nn.Parameter(original_pos_embed)
                    
                    # 构建一个上下文管理器用于后续恢复
                    import contextlib
                    @contextlib.contextmanager
                    def temp_pos_embed():
                        try:
                            yield
                        finally:
                            restore_pos_embed()
                    
                    # 使用该上下文管理器
                    with temp_pos_embed():
                        # 调整输入图像大小
                        if original_size[0] != image_size or original_size[1] != image_size:
                            print(f"调整输入图像从 {original_size} 到 {image_size}x{image_size}")
                            images = F.interpolate(
                                images, 
                                size=(image_size, image_size), 
                                mode='bilinear', 
                                align_corners=False
                            )
                        
                        # 使用混合精度计算
                        with torch.cuda.amp.autocast():
                            # 图像编码
                            image_embeddings = self.image_encoder(images)
                            
                            # # 初始化提示编码
                            # if points is None:
                            #     # 如果没有提供点提示，使用默认的中心点提示
                            #     coords = torch.tensor([[[image_size//2, image_size//2]]], 
                            #                         device=images.device, dtype=torch.float32)
                            #     labels = torch.tensor([[1]], device=images.device, dtype=torch.float32)
                            #     points = (coords, labels)
                            
                            # 获取提示编码
                            sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                                points=points,
                                boxes=boxes,
                                masks=masks,
                            )
                            
                            # 使用掩码解码器
                            low_res_masks, iou_predictions = self.mask_decoder(
                                image_embeddings=image_embeddings,
                                image_pe=self.sam.prompt_encoder.get_dense_pe(),
                                sparse_prompt_embeddings=sparse_embeddings,
                                dense_prompt_embeddings=dense_embeddings,
                                multimask_output=multimask_output,
                            )
                    
                    # 此时位置嵌入已经自动恢复
                else:
                    # 如果不需要调整位置嵌入，正常处理
                    # 调整输入图像大小
                    if original_size[0] != image_size or original_size[1] != image_size:
                        print(f"调整输入图像从 {original_size} 到 {image_size}x{image_size}")
                        images = F.interpolate(
                            images, 
                            size=(image_size, image_size), 
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    # 使用混合精度计算
                    with torch.cuda.amp.autocast():
                        # 图像编码
                        image_embeddings = self.image_encoder(images)
                        
                        # 初始化提示编码
                        # if points is None:
                        #     # 如果没有提供点提示，使用默认的中心点提示
                        #     coords = torch.tensor([[[image_size//2, image_size//2]]], 
                        #                         device=images.device, dtype=torch.float32)
                        #     labels = torch.tensor([[1]], device=images.device, dtype=torch.float32)
                        #     points = (coords, labels)
                        
                        # 获取提示编码
                        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                            points=points,
                            boxes=boxes,
                            masks=masks,
                        )
                        
                        # 使用掩码解码器
                        low_res_masks, iou_predictions = self.mask_decoder(
                            image_embeddings=image_embeddings,
                            image_pe=self.sam.prompt_encoder.get_dense_pe(),
                            sparse_prompt_embeddings=sparse_embeddings,
                            dense_prompt_embeddings=dense_embeddings,
                            multimask_output=multimask_output,
                        )
            else:
                # 没有位置嵌入，直接处理
                # 调整输入图像大小
                if original_size[0] != image_size or original_size[1] != image_size:
                    print(f"调整输入图像从 {original_size} 到 {image_size}x{image_size}")
                    images = F.interpolate(
                        images, 
                        size=(image_size, image_size), 
                        mode='bilinear', 
                        align_corners=False
                    )
                
                # 使用混合精度计算
                with torch.cuda.amp.autocast():
                    # 图像编码
                    image_embeddings = self.image_encoder(images)
                    
                    # 初始化提示编码
                    # if points is None:
                    #     # 如果没有提供点提示，使用默认的中心点提示
                    #     coords = torch.tensor([[[image_size//2, image_size//2]]], 
                    #                         device=images.device, dtype=torch.float32)
                    #     labels = torch.tensor([[1]], device=images.device, dtype=torch.float32)
                    #     points = (coords, labels)
                    
                    # 获取提示编码
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=points,
                        boxes=boxes,
                        masks=masks,
                    )
                    
                    # 使用掩码解码器
                    low_res_masks, iou_predictions = self.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=self.sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=multimask_output,
                    )
        else:
            # 原始尺寸就是1024，不需要调整
            # 使用混合精度计算
            with torch.cuda.amp.autocast():
                # 图像编码
                image_embeddings = self.image_encoder(images)
                
                # 初始化提示编码
                # if points is None:
                #     # 如果没有提供点提示，使用默认的中心点提示
                #     coords = torch.tensor([[[image_size//2, image_size//2]]], 
                #                         device=images.device, dtype=torch.float32)
                #     labels = torch.tensor([[1]], device=images.device, dtype=torch.float32)
                #     points = (coords, labels)
                
                # 获取提示编码
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                    points=points,
                    boxes=boxes,
                    masks=masks,
                )
                
                # 使用掩码解码器
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=self.sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
        
        # 确保输出mask是4维张量
        if low_res_masks.dim() == 2:
            low_res_masks = low_res_masks.unsqueeze(0).unsqueeze(0)
        elif low_res_masks.dim() == 3:
            low_res_masks = low_res_masks.unsqueeze(1)
        
        # 清理不需要的中间变量
        del sparse_embeddings, dense_embeddings, image_embeddings
        torch.cuda.empty_cache()
        
        # 返回预测结果
        return {
            'masks': low_res_masks,
            'iou_predictions': iou_predictions,
            'low_res_masks': low_res_masks
        }


    # def forward(self, x: Tensor) -> Tensor:
    #     return self.lora_vit(x)



def test_lora_layer_initialization():
    # 创建模拟SAM模型
    class MockSam:
        def __init__(self):
            self.image_encoder = type('', (), {'blocks': [type('', (), {'attn': None})]})()
    
    # 测试正常情况
    sam = MockSam()
    try:
        layer = Linear(768, 2304, sam_model=sam, r=4)
        assert True
    except TypeError:
        pytest.fail("初始化失败")
    
    # 测试缺少sam_model的情况
    with pytest.raises(TypeError):
        Linear(768, 2304, r=4)  # 缺少sam_model应报错
