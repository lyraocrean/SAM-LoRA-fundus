"""
train_lora_sam.py
- 使用插入到图像编码器QKV中的LoRA适配器训练SAM-ViT-B（参见add_lora.py）。
- 默认情况下，基础SAM处于冻结状态；仅优化LoRA参数（如果未冻结，也可选择优化掩码解码器）。
- 使用Albumentations进行数据增强、混合精度训练+梯度累积、余弦学习率调度、早停机制、TensorBoard日志记录，以及在result/runs目录下为每次运行生成指标CSV/JSON文件。
- CLAHE预处理是可选的（默认仅用于验证路径）以节省内存；训练过程会跳过CLAHE。
- 数据集：当is_fives=True时，需要FIVES风格的文件夹（包含Original / Ground truth）。
- 入口：配置CONFIG，确保checkpoint_path指向sam_vit_b.pth，然后运行main()。
"""

import sys
import os
import time
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import traceback  # 添加traceback库以打印详细错误
import math
import cv2
import csv
import gc
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn.functional as F
from kornia.filters import sobel
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import shutil
import numpy as np
from contextlib import nullcontext
from swanlab.plugin.notification import LarkCallback

# 添加SAM相关导入
from segment_anything import sam_model_registry
from add_lora import LoRA_Sam
from torch.utils.tensorboard import SummaryWriter
from segment_anything.modeling import Sam
from prompt_generation import (
    generate_prompts_from_mask,
    generate_box_prompts_from_mask,
    generate_terminal_box_prompts_from_mask,
    encode_boxes_as_sam_points,
)

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as transforms
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from torch.cuda.amp import autocast, GradScaler
from metrics_utils import (
    compute_segmentation_metrics,
    DEFAULT_THRESHOLDS,
    save_quick_visual_grid,
)


# ==== Config (集中所有可调参数) ====
CONFIG = {
    # 路径与运行名
    "output_root": "result/runs",
    "run_name": "fives",
    # 训练超参
    "num_epochs": 100,
    "batch_size": 2,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "image_size": 1024,
    "accumulation_steps": 2,
    "use_prompts": True,
    "prompt_mode": "pos_neg",
    "prompt_edge_distance": 4,
    "prompt_neg_min_distance": 5,
    "prompt_box_min_count": 3,
    "prompt_box_max_count": 4,
    "prompt_box_size": 96,
    "prompt_box_min_component_area": 50,
    "prompt_box_min_center_distance": 40,
    "prompt_box_terminal_radius": 8,
    # 数据路径（通过命令行参数传入，见 main()）
    "checkpoint_path": None,
    "data_path": None,
    "test_path": None,
    # resume 相关
    "resume": None,
    # 通知（可选，通过环境变量配置）
    "wechat_webhook": os.getenv("WECHAT_WEBHOOK"),
    "smtp_host": os.getenv("SMTP_HOST"),
    "smtp_user": os.getenv("SMTP_USER"),
    "smtp_pass": os.getenv("SMTP_PASS"),
    "mail_to": os.getenv("MAIL_TO"),
    "lark_webhook": os.getenv("LARK_WEBHOOK"),
    "lark_secret": os.getenv("LARK_SECRET"),

}

# 覆盖配置：补充评估阈值与可视化开关
CONFIG = {
    **CONFIG,
    "eval_thresholds": [round(x, 2) for x in np.linspace(0.05, 0.95, 19)],
    "enable_quick_vis": True,
    "quick_vis_samples": 16,
    "quick_vis_seed": 42,
}


class EarlyStopping:
    """早停机制，避免过拟合"""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss)
            self.counter = 0
        return False

    def save_checkpoint(self, val_loss):
        if self.verbose:
            print(f'验证损失降低 ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
        
    def reset(self):
        self.counter = 0

class MedicalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        mask_transform=None,
        is_fives=False,
        use_prompts=False,
        prompt_mode="pos_neg",
        prompt_edge_distance=4,
        prompt_neg_min_distance=5,
        prompt_box_min_count=3,
        prompt_box_max_count=4,
        prompt_box_size=96,
        prompt_box_min_component_area=50,
        prompt_box_min_center_distance=40,
        prompt_box_terminal_radius=8,
    ):
        self.data_dir = data_dir
        self.is_fives = is_fives
        self.transform = transform
        self.mask_transform = mask_transform if mask_transform is not None else transform
        self.use_prompts = use_prompts
        self.prompt_mode = prompt_mode
        self.prompt_edge_distance = prompt_edge_distance
        self.prompt_neg_min_distance = prompt_neg_min_distance
        self.prompt_box_min_count = prompt_box_min_count
        self.prompt_box_max_count = prompt_box_max_count
        self.prompt_box_size = prompt_box_size
        self.prompt_box_min_component_area = prompt_box_min_component_area
        self.prompt_box_min_center_distance = prompt_box_min_center_distance
        self.prompt_box_terminal_radius = prompt_box_terminal_radius
        
        if is_fives:
            # FIVES数据集结构
            self.image_dir = os.path.join(data_dir, "Original")
            self.mask_dir = os.path.join(data_dir, "Ground truth")
            
            # 添加文件检查
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                     if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
            
            # 验证每个图像都有对应的mask
            self.valid_pairs = []
            for img_file in self.image_files:
                base_name = os.path.splitext(img_file)[0]
                
                # 查找对应的mask文件 (文件名完全匹配)
                mask_file = img_file  # FIVES数据集中图像和标签文件名一致
                mask_path = os.path.join(self.mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    # 跳过全零的空mask
                    try:
                        mask_img = Image.open(mask_path).convert("L")
                        mask_np = np.array(mask_img)
                        if mask_np.max() == 0:
                            print(f"跳过空mask: {mask_file}")
                            continue
                    except Exception as e:
                        print(f"读取mask失败 {mask_file}: {e}")
                        continue
                    self.valid_pairs.append((img_file, mask_file))
                else:
                    print(f"警告：图像 {img_file} 没有找到对应的mask文件")
        else:
            # 原始DRIVE数据集结构
            self.image_dir = os.path.join(data_dir, "images")
            self.mask_dir = os.path.join(data_dir, "masks")
            
            # 添加文件检查
            self.image_files = sorted([f for f in os.listdir(self.image_dir) 
                                     if f.endswith(('.tif', '.png', '.jpg', '.jpeg'))])
            
            # 验证每个图像都有对应的mask
            self.valid_pairs = []
            for img_file in self.image_files:
                base_name = os.path.splitext(img_file)[0]
                number_part = base_name.split('_')[0]
                
                # 查找对应的mask文件
                mask_found = False
                for mask_file in os.listdir(self.mask_dir):
                    if mask_file.split('_')[0] == number_part:
                        self.valid_pairs.append((img_file, mask_file))
                        mask_found = True
                        break
                
                if not mask_found:
                    print(f"警告：图像 {img_file} 没有找到对应的mask文件")
        
        print(f"找到 {len(self.valid_pairs)} 个有效的图像-mask对")

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        max_retries = 3
        for retry in range(max_retries):
            try:
                img_file, mask_file = self.valid_pairs[idx]
                
                # 加载图像
                img_path = os.path.join(self.image_dir, img_file)
                mask_path = os.path.join(self.mask_dir, mask_file)
                
                # 使用PIL加载图像
                img = Image.open(img_path).convert("RGB")
                mask = Image.open(mask_path).convert("L")
                
                # 将PIL图像转换为numpy数组，用于Albumentations
                img_np = np.array(img)
                mask_np = np.array(mask)
                
                # 确保掩码是二值的
                mask_np = (mask_np > 127).astype(np.uint8) * 255
                
                # 应用Albumentations增强（正确地使用命名参数）
                if self.transform:
                    transformed = self.transform(image=img_np, mask=mask_np)
                    img_transformed = transformed['image']
                    mask_transformed = transformed['mask']
                    if not isinstance(img_transformed, torch.Tensor):
                        img_transformed = torch.from_numpy(img_transformed.transpose(2, 0, 1)).float() / 255.0
                    if not isinstance(mask_transformed, torch.Tensor):
                        mask_transformed = torch.from_numpy(mask_transformed).unsqueeze(0).float() / 255.0
                else:
                    img_transformed = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
                    mask_transformed = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0

                if self.use_prompts:
                    # 将mask转为二值numpy，用于生成点
                    mask_for_prompt = mask_transformed
                    if isinstance(mask_for_prompt, torch.Tensor):
                        mask_np_prompt = mask_for_prompt.squeeze().detach().cpu().numpy()
                    else:
                        mask_np_prompt = np.array(mask_for_prompt)
                    mask_np_prompt = (mask_np_prompt > 0.5).astype(np.uint8)
                    try:
                        if self.prompt_mode in ("box_only", "box_terminal"):
                            if self.prompt_mode == "box_terminal":
                                box_prompts = generate_terminal_box_prompts_from_mask(
                                    mask_np_prompt,
                                    min_boxes=self.prompt_box_min_count,
                                    max_boxes=self.prompt_box_max_count,
                                    box_size=self.prompt_box_size,
                                    min_component_area=self.prompt_box_min_component_area,
                                    min_center_distance=self.prompt_box_min_center_distance,
                                    terminal_radius=self.prompt_box_terminal_radius,
                                    seed=idx,
                                )
                            else:
                                box_prompts = generate_box_prompts_from_mask(
                                    mask_np_prompt,
                                    min_boxes=self.prompt_box_min_count,
                                    max_boxes=self.prompt_box_max_count,
                                    box_size=self.prompt_box_size,
                                    min_component_area=self.prompt_box_min_component_area,
                                    min_center_distance=self.prompt_box_min_center_distance,
                                    seed=idx,
                                )
                            coords_xy, labels = encode_boxes_as_sam_points(box_prompts.boxes_xyxy)
                        else:
                            prompts = generate_prompts_from_mask(
                                mask_np_prompt,
                                edge_distance_thresh=self.prompt_edge_distance,
                                negative_min_distance=self.prompt_neg_min_distance,
                                seed=idx,
                            )
                            pos_points = [prompts.center] + prompts.edge
                            neg_points = prompts.negative if self.prompt_mode == "pos_neg" else []
                            all_points = pos_points + neg_points
                            labels = [1] * len(pos_points) + [0] * len(neg_points)
                            coords_xy = [(c[1], c[0]) for c in all_points]

                        point_coords = torch.tensor(coords_xy, dtype=torch.float32)
                        point_labels = torch.tensor(labels, dtype=torch.int64)
                    except Exception as e:
                        print(f"Prompt generation failed at idx {idx}: {e}")
                        point_coords = torch.empty((0, 2), dtype=torch.float32)
                        point_labels = torch.empty((0,), dtype=torch.int64)

                    return {
                        "image": img_transformed,
                        "mask": mask_transformed,
                        "point_coords": point_coords,
                        "point_labels": point_labels,
                    }

                return img_transformed, mask_transformed
                
            except Exception as e:
                print(f"处理索引 {idx} 重试 {retry+1}/{max_retries}: {str(e)}")
                if retry == max_retries - 1:
                    # 如果是最后一次重试，返回一个默认值或尝试下一个样本
                    print(f"无法加载索引 {idx}，跳过该样本")
                    # 返回一个空白图像和掩码
                    blank_image = torch.zeros(3, 1024, 1024)
                    blank_mask = torch.zeros(1, 1024, 1024)
                    if self.use_prompts:
                        return {
                            "image": blank_image,
                            "mask": blank_mask,
                            "point_coords": torch.empty((0, 2), dtype=torch.float32),
                            "point_labels": torch.empty((0,), dtype=torch.int64),
                        }
                    return blank_image, blank_mask


def collate_with_prompts(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, dict):
        images = torch.stack([b["image"] for b in batch], dim=0)
        masks = torch.stack([b["mask"] for b in batch], dim=0)

        coords_list = [b.get("point_coords") for b in batch]
        labels_list = [b.get("point_labels") for b in batch]
        max_points = max((c.shape[0] for c in coords_list), default=0)

        if max_points == 0:
            point_coords = torch.empty((len(batch), 0, 2), dtype=torch.float32)
            point_labels = torch.empty((len(batch), 0), dtype=torch.int64)
        else:
            point_coords = torch.zeros((len(batch), max_points, 2), dtype=torch.float32)
            point_labels = torch.full((len(batch), max_points), -1, dtype=torch.int64)
            for i, (coords, labels) in enumerate(zip(coords_list, labels_list)):
                if coords is None or coords.numel() == 0:
                    continue
                n = coords.shape[0]
                point_coords[i, :n] = coords
                point_labels[i, :n] = labels

        return {
            "image": images,
            "mask": masks,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    return default_collate(batch)


def compute_dice_loss(pred, target, smooth=1.0):
    """计算Dice损失，添加focal机制和类平衡权重"""
    # 确保输入是浮点类型
    pred = pred.float()
    target = target.float()
    
    # 添加数值稳定性检查
    pred = torch.clamp(pred, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)
    
    # 获取正样本和负样本数量
    num_pos = torch.sum(target)
    num_neg = torch.sum(1.0 - target)
    
    # 计算类权重
    pos_weight = torch.where(num_pos > 0, 
                           num_neg / (num_pos + 1e-6), 
                           torch.tensor(1.0, device=target.device))
    pos_weight = torch.clamp(pos_weight, 0.5, 50.0).float()
    
    # 应用Focal调整因子
    gamma = 2.0
    pt = pred * target + (1 - pred) * (1 - target)
    focal_weight = (1 - pt) ** gamma
    
    # 计算带权重的Dice
    intersection = torch.sum(pred * target * focal_weight)
    union = torch.sum(pred * focal_weight) + torch.sum(target * focal_weight * pos_weight)
    
    # 添加数值稳定性检查
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (union + smooth)
    dice_loss = torch.clamp(dice_loss, 0.0, 1.0)
    
    return dice_loss

def compute_bce_loss(pred, target, smooth=1.0):
    """计算带类平衡的BCE损失"""
    # 确保输入是浮点类型
    pred = pred.float()
    target = target.float()
    
    # 添加数值稳定性检查
    pred = torch.clamp(pred, -10.0, 10.0)  # 限制logits范围
    target = torch.clamp(target, 0.0, 1.0)
    
    # 计算正负样本比例
    num_pos = torch.sum(target)
    num_neg = torch.sum(1.0 - target)
    total = num_pos + num_neg
    
    # 动态计算类别权重
    pos_weight = torch.where(num_pos > 0, 
                            num_neg / (num_pos + 1e-6), 
                            torch.tensor(1.0, device=target.device))
    
    # 限制pos_weight的范围，并确保是浮点类型
    pos_weight = torch.clamp(pos_weight, 0.5, 50.0).float()
    
    # 使用带权重的BCE
    bce = F.binary_cross_entropy_with_logits(
        pred, target, 
        pos_weight=pos_weight,
        reduction='none'
    )
    
    # 应用Focal调整因子
    gamma = 2.0
    pt = torch.exp(-bce)
    focal_bce = (1 - pt) ** gamma * bce
    
    # 添加数值稳定性检查
    focal_bce = torch.clamp(focal_bce, 0.0, 10.0)
    
    return focal_bce.mean()

def focal_tversky_loss(logits, target, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
    target = target.float()
    probs = torch.sigmoid(logits).clamp(1e-6, 1-1e-6)
    tp = (probs * target).sum()
    fp = (probs * (1 - target)).sum()
    fn = ((1 - probs) * target).sum()
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return (1 - tversky) ** gamma


def combined_loss(pred, target, bce_weight=0.7, dice_weight=0.3, ft_weight=0.2, ft_alpha=0.7, ft_beta=0.3, ft_gamma=1.33):
    """组合BCE与Dice损失，添加动态权重调整"""
    # 确保输入是浮点类型
    pred = pred.float()
    target = target.float()
    
    bce = compute_bce_loss(pred, target)
    
    
    # 计算sigmoid并应用阈值以避免梯度问题
    probs = torch.sigmoid(pred)
    dice = compute_dice_loss(probs, target)
    ft = focal_tversky_loss(pred, target, alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
    loss = bce_weight * bce + dice_weight * dice + ft_weight * ft
    
    # 动态调整损失权重
    if bce > 0.8:  # BCE较高时提高Dice权重
        dice_weight = min(0.6, dice_weight * 1.2)
        bce_weight = 1.0 - dice_weight
    
    # 组合损失
    loss = bce_weight * bce + dice_weight * dice
    
    # 添加数值稳定性检查
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"警告: 损失值为 {loss}, 使用默认值 1.0")
        loss = torch.tensor(1.0, device=loss.device)
    
    return loss

def validate_dataset(dataset):
    print("正在验证数据集...")
    try:
        # 检查第一个样本
        sample = dataset[0]
        if isinstance(sample, dict):
            images, masks = sample["image"], sample["mask"]
        else:
            images, masks = sample
        print(f"数据集验证成功! 形状: 图像 {images.shape}, 掩码 {masks.shape}")
        return True
    except Exception as e:
        print(f"数据集验证失败: {str(e)}")
        return False

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"GPU缓存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")

def preprocess_images(images, use_clahe=False):
    """使用CLAHE增强图像对比度"""
    if not use_clahe:
        return images

    device = images.device
    channels = images.shape[1]
    mean = torch.zeros(1, channels, 1, 1, device=device)
    std = torch.ones(1, channels, 1, 1, device=device)
    if channels == 3:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    
    # 反归一化到[0,1]后在CPU上执行CLAHE
    images_cpu = (images * std + mean).detach().cpu()
    processed = []

    for i in range(images_cpu.size(0)):
        img = images_cpu[i].permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if img.shape[2] == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            lab_planes = list(cv2.split(lab))
            lab_planes[0] = clahe.apply(lab_planes[0])
            lab = cv2.merge(lab_planes)
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            img = clahe.apply(img if img.ndim == 2 else img[..., 0])
            img = np.expand_dims(img, axis=-1)

        img_tensor = torch.from_numpy(img).float() / 255.0
        processed.append(img_tensor.permute(2, 0, 1))

    processed_images = torch.stack(processed, dim=0)
    processed_images = (processed_images - mean.cpu()) / std.cpu()
    return processed_images.to(device)

def get_transforms(is_train=True, image_size=1024):
    """获取增强的数据增强转换"""
    if is_train:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent=(-0.0625, 0.0625),
                scale=(0.8, 1.2),
                rotate=(-45, 45),
                p=0.7
            ),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.6),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.6),
                A.OpticalDistortion(distort_limit=1.0, p=0.6),
            ], p=0.5),
            # 局部过曝/高光斑
            A.RandomSunFlare(
                flare_roi=(0.1, 0.1, 0.9, 0.9),
                angle_range=(0, 1),
                num_flare_circles_range=(1, 3),
                src_radius=80, src_color=(255, 255, 255),
                p=0.35
            ),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
            # gamma 抖动
            A.RandomGamma(gamma_limit=(70, 130), p=0.4),
            A.OneOf([
                A.GaussNoise(std_range=(0.012, 0.028), p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
                A.MotionBlur(blur_limit=3, p=0.5),
            ], p=0.5),
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(8, 32),
                hole_width_range=(8, 32),
                fill=0,
                p=0.5
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def train_one_epoch(model, train_loader, optimizer, device, scaler, epoch, accum_steps=4, eval_thresholds=DEFAULT_THRESHOLDS):
    """
    训练一个 epoch 的模型，支持梯度累积和混合精度
    
    参数:
        model: 模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        device: 设备
        scaler: 混合精度缩放器
        epoch: 当前 epoch 编号
        accum_steps: 梯度累积步数
    返回:
        平均损失, Dice系数, IoU系数, 训练指标字典
    """
    model.train()
    total_loss = 0
    processed_samples = 0
    optimizer.zero_grad()
    
    # 初始化训练指标字典
    train_metrics = {
        'dice': 0.0,
        'iou': 0.0,
        'precision': 0.0,
        'recall': 0.0,
        'f1': 0.0,
        'accuracy': 0.0,
        'specificity': 0.0
    }
    
    # 显示epoch进度
    print(f"\nEpoch {epoch+1} 训练中...", end="")
    
    for batch_idx, batch in enumerate(train_loader):
        # 将数据移到正确的设备上
        if isinstance(batch, dict):
            images = batch["image"]
            masks = batch["mask"]
            point_coords = batch.get("point_coords")
            point_labels = batch.get("point_labels")
        else:
            images, masks = batch
            point_coords = None
            point_labels = None

        if point_coords is not None and point_coords.numel() == 0:
            point_coords = None
        if point_labels is not None and point_labels.numel() == 0:
            point_labels = None

        images = images.to(device)
        masks = masks.to(device)
        if point_coords is not None:
            point_coords = point_coords.to(device)
        if point_labels is not None:
            point_labels = point_labels.to(device)

        if point_coords is not None and point_labels is not None:
            points = (point_coords, point_labels)
        else:
            points = None
        # 确保masks有正确的维度格式
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        
        # 预处理图像
        processed_images = preprocess_images(images)
        
        # 使用自动混合精度训练
        with torch.cuda.amp.autocast():
            # 前向传播
            outputs = model(
                processed_images,
                points=points,
                multimask_output=False,
            )
            outputs = outputs["masks"]
            
            # 确保掩码尺寸匹配
            if outputs.shape[-2:] != masks.shape[-2:]:
                outputs = F.interpolate(
                    outputs,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            
            # 计算损失
            loss = combined_loss(outputs, masks)
            loss = loss / accum_steps
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        # 计算指标（统一口径：sigmoid->thresholds->per-image平均）
        with torch.no_grad():
            batch_metrics, _ = compute_segmentation_metrics(
                outputs,
                masks,
                thresholds=eval_thresholds,
                reduction="mean",
            )
        
        # 累计统计
        batch_size = images.size(0)
        total_loss += loss.item() * accum_steps
        for k in train_metrics.keys():
            train_metrics[k] += batch_metrics[k] * batch_size
        processed_samples += batch_size
        
        # 显示进度
        if (batch_idx + 1) % 10 == 0:
            print(".", end="", flush=True)
    
    print("完成")
    
    # 计算平均指标
    avg_loss = total_loss / len(train_loader)
    avg_dice = train_metrics['dice'] / processed_samples
    avg_iou = train_metrics['iou'] / processed_samples
    for k in train_metrics.keys():
        train_metrics[k] = train_metrics[k] / processed_samples
    
    return avg_loss, avg_dice, avg_iou, train_metrics

def validate(model, val_loader, device, scaler=None, eval_thresholds=DEFAULT_THRESHOLDS):
    """
    验证一个 epoch，支持多阈值扫描，返回最佳阈值及对应指标。
    """
    model.eval()
    val_loss = 0.0
    processed_samples = 0

    # 准备阈值与累积容器
    thresholds = tuple(eval_thresholds) if eval_thresholds else DEFAULT_THRESHOLDS
    metric_keys = ["dice", "iou", "precision", "recall", "f1", "accuracy", "specificity"]
    sum_metrics = {k: torch.zeros(len(thresholds), device=device) for k in metric_keys}

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if isinstance(batch, (list, tuple)):
                images, masks = batch[0], batch[1]
                point_coords = None
                point_labels = None
            else:
                images, masks = batch["image"], batch["mask"]
                point_coords = batch.get("point_coords")
                point_labels = batch.get("point_labels")
            if point_coords is not None and point_coords.numel() == 0:
                point_coords = None
            if point_labels is not None and point_labels.numel() == 0:
                point_labels = None
            images = images.to(device)
            masks = masks.to(device)
            if point_coords is not None:
                point_coords = point_coords.to(device)
            if point_labels is not None:
                point_labels = point_labels.to(device)

            if point_coords is not None and point_labels is not None:
                points = (point_coords, point_labels)
            else:
                points = None
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)

            with autocast(enabled=scaler is not None):
                outputs = model(
                    images,
                    points=points,
                    multimask_output=False,
                )["masks"]
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(
                        outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                loss = combined_loss(outputs, masks)
            val_loss += loss.item()

            batch_metrics, _ = compute_segmentation_metrics(
                outputs, masks, thresholds=thresholds, reduction="per_threshold"
            )
            batch_size = images.size(0)
            for k in metric_keys:
                sum_metrics[k] += batch_metrics[k].to(device) * batch_size
            processed_samples += batch_size

            if (batch_idx + 1) % 5 == 0:
                print(".", end="", flush=True)

    print("完成")

    val_loss /= max(len(val_loader), 1)
    avg_metrics = {k: (v / max(processed_samples, 1)).cpu().tolist() for k, v in sum_metrics.items()}
    best_idx = int(torch.tensor(avg_metrics["f1"]).argmax())
    best_thresh = thresholds[best_idx]
    best_metrics = {k: avg_metrics[k][best_idx] for k in metric_keys}

    return val_loss, best_metrics, best_thresh, avg_metrics




def build_lark():
    url = CONFIG.get("lark_webhook")
    secret = CONFIG.get("lark_secret")
    if not url:
        return None
    try:
        return LarkCallback(webhook_url=url, secret=secret)
    except Exception as e:
        print(f"init LarkCallback failed: {e}")
        return None

def lark_send(cb: LarkCallback, text: str):
    if cb is None:
        return
    try:
        cb.send_msg(content=text)
    except Exception as e:
        print(f"Lark send failed: {e}")


def main():
    try:
        # 禁用Albumentations更新检查
        os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
        
        # 调用前面配置好的路径
        cfg = CONFIG
        lark_cb = build_lark()

        checkpoint_path = cfg["checkpoint_path"]
        data_path = cfg["data_path"]
        test_path = cfg["test_path"]
        
        # 训练参数优化
        num_epochs = cfg["num_epochs"]
        batch_size = cfg["batch_size"]
        learning_rate = cfg["learning_rate"]
        weight_decay = cfg["weight_decay"]
        image_size = cfg["image_size"]
        accumulation_steps = cfg["accumulation_steps"]
        eval_thresholds = cfg.get("eval_thresholds", [0.5])
        
        # 输出设置
        timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        run_dir = os.path.join(cfg["output_root"], f"{timestamp}-{cfg['run_name']}")
        ckpt_dir = os.path.join(run_dir, "checkpoints")
        tb_dir = os.path.join(run_dir, "tb")
        metrics_dir = os.path.join(run_dir, "metrics")
        logs_dir = os.path.join(run_dir, "logs")
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(tb_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)

        # 记录训练开始时间和当前参数
        start_time = time.time()
        start_datetime = datetime.now()
        log_path = os.path.join(logs_dir, "training_log.txt")
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Run start: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Run dir: {run_dir}\n")
            log_file.write("Config:\n")
            log_file.write(json.dumps(cfg, indent=2, ensure_ascii=False))
            log_file.write("\n\n")

        # 添加设备检查
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cpu":
            print("警告: 使用CPU训练,速度可能较慢")
        
        # 加载SAM模型
        print("正在加载SAM模型...")
        sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        sam.to(device)
        
        # 创建LoRA模型实例
        print("正在初始化LoRA模型...")
        lora_sam = LoRA_Sam(sam, 4,lora_alpha=9, lora_dropout=0.1)
        lora_sam.to(device)

        # 准备数据加载器
        print("正在准备数据加载器...")
        train_transform = get_transforms(is_train=True, image_size=image_size)
        val_transform = get_transforms(is_train=False, image_size=image_size)
        
        dataset = MedicalDataset(
            data_path,
            transform=train_transform,
            is_fives=True,
            use_prompts=cfg.get("use_prompts", False),
            prompt_mode=cfg.get("prompt_mode", "pos_neg"),
            prompt_edge_distance=cfg.get("prompt_edge_distance", 4),
            prompt_neg_min_distance=cfg.get("prompt_neg_min_distance", 5),
            prompt_box_min_count=cfg.get("prompt_box_min_count", 3),
            prompt_box_max_count=cfg.get("prompt_box_max_count", 4),
            prompt_box_size=cfg.get("prompt_box_size", 96),
            prompt_box_min_component_area=cfg.get("prompt_box_min_component_area", 50),
            prompt_box_min_center_distance=cfg.get("prompt_box_min_center_distance", 40),
            prompt_box_terminal_radius=cfg.get("prompt_box_terminal_radius", 8),
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_with_prompts,
        )
        
        val_dataset = MedicalDataset(
            test_path,
            transform=val_transform,
            is_fives=True,
            use_prompts=cfg.get("use_prompts", False),
            prompt_mode=cfg.get("prompt_mode", "pos_neg"),
            prompt_edge_distance=cfg.get("prompt_edge_distance", 4),
            prompt_neg_min_distance=cfg.get("prompt_neg_min_distance", 5),
            prompt_box_min_count=cfg.get("prompt_box_min_count", 3),
            prompt_box_max_count=cfg.get("prompt_box_max_count", 4),
            prompt_box_size=cfg.get("prompt_box_size", 96),
            prompt_box_min_component_area=cfg.get("prompt_box_min_component_area", 50),
            prompt_box_min_center_distance=cfg.get("prompt_box_min_center_distance", 40),
            prompt_box_terminal_radius=cfg.get("prompt_box_terminal_radius", 8),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_with_prompts,
        )
        
        # 验证数据集
        print("正在验证数据集...")
        if not validate_dataset(dataset):
            print("数据集验证失败,请检查数据")
            return
        
        # 设置优化器
        print("正在设置优化器...")
        # optimizer = torch.optim.AdamW(
        #     lora_sam.parameters(),
        #     lr=learning_rate,
        #     weight_decay=weight_decay
        # )

        trainable = [p for p in lora_sam.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)

        
        # 设置学习率调度器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10,
            T_mult=2,
            eta_min=learning_rate/100
        )
        
        # 使用混合精度训练的scaler
        scaler = torch.cuda.amp.GradScaler()
        
        # 创建早停机制
        early_stopping = EarlyStopping(patience=5, verbose=True, delta=0.001)
        
        # 创建TensorBoard写入器
        writer = SummaryWriter(tb_dir)

        # 初始化训练状态
        start_epoch = 0
        best_val_loss = float('inf')
        best_val_iou = 0
        best_model_path = os.path.join(ckpt_dir, "best-iou.pth")
        
        # 恢复训练（如果指定了checkpoint）
        resume_path = cfg["resume"]
        if resume_path:
            ckpt = torch.load(resume_path, map_location=device)
            lora_sam.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            scaler.load_state_dict(ckpt.get("scaler_state_dict", scaler.state_dict()))
            best_val_iou = ckpt.get("best_val_iou", 0)
            best_val_loss = ckpt.get("best_val_loss", float('inf'))
            start_epoch = ckpt.get("epoch", -1) + 1
            print(f"Resumed from {resume_path}, start_epoch={start_epoch}")
        
        # 训练循环
        print("\n开始训练...")
        print("\nEpoch\t训练损失\t训练Dice\t训练IoU\t训练准确率\t训练召回率\t训练F1\t验证损失\t验证Dice\t验证IoU\t验证准确率\t验证召回率\t验证F1")
        print("-" * 120)
        
        for epoch in range(start_epoch, num_epochs):
            # 训练一个epoch
            train_loss, train_dice, train_iou, train_metrics = train_one_epoch(
                lora_sam, dataloader, optimizer, device, scaler, epoch, accumulation_steps, eval_thresholds
            )
            
            # 验证
            val_loss, val_metrics, best_thresh, val_metrics_all = validate(
                lora_sam, val_loader, device, scaler, eval_thresholds)       

            val_dice = val_metrics['dice']
            val_iou = val_metrics['iou']
            
            # 更新学习率
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # 打印训练和验证结果
            print(f"{epoch+1:3d}\t{train_loss:.4f}\t{train_dice:.4f}\t{train_iou:.4f}\t"
                f"{train_metrics['accuracy']:.4f}\t{train_metrics['recall']:.4f}\t{train_metrics['f1']:.4f}\t"
                f"{val_loss:.4f}\t{val_dice:.4f}\t{val_iou:.4f}\t"
                f"{val_metrics['accuracy']:.4f}\t{val_metrics['recall']:.4f}\t{val_metrics['f1']:.4f}")
            
            # 记录指标到TensorBoard
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Dice/train', train_dice, epoch)
            writer.add_scalar('Dice/val', val_dice, epoch)
            writer.add_scalar('IoU/train', train_iou, epoch)
            writer.add_scalar('IoU/val', val_iou, epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            writer.add_scalar('Recall/train', train_metrics['recall'], epoch)
            writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
            writer.add_scalar('F1/train', train_metrics['f1'], epoch)
            writer.add_scalar('F1/val', val_metrics['f1'], epoch)
            writer.add_scalar('LR', current_lr, epoch)
            
            # 保存 checkpoint（含优化器/调度器/Scaler）
            ckpt = {
                "epoch": epoch,
                "model_state_dict": lora_sam.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "val_loss": val_loss,
                "val_iou": val_iou,
                "val_metrics": val_metrics,
                "best_val_iou": best_val_iou,
                "best_val_loss": best_val_loss,
                "config": cfg,
            }
            # 保存当前 epoch
            torch.save(ckpt, os.path.join(ckpt_dir, f"epoch-{epoch+1}.pth"))
            # 保存 best
            if val_iou > best_val_iou:
                best_val_iou = val_iou
                best_val_loss = val_loss
                print(f"\n保存最佳模型 - 验证IoU: {val_iou:.4f}, 验证F1: {val_metrics['f1']:.4f}")
                torch.save(ckpt, os.path.join(ckpt_dir, "best-iou.pth"))
            if val_iou > 0.95:
                lark_send(lark_cb, f"Val IoU passed 0.95: {val_iou:.4f} (epoch {epoch+1})")

                
            
            # 记录结构化指标
            metrics_row = {
                "epoch": epoch+1,
                "train_loss": train_loss,
                "train_dice": train_dice,
                "train_iou": train_iou,
                "val_loss": val_loss,
                "val_dice": val_metrics["dice"],
                "val_iou": val_metrics["iou"],
                "val_acc": val_metrics["accuracy"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "lr": current_lr,
            }
            csv_path = os.path.join(metrics_dir, "metrics.csv")
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                writer_csv = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
                if write_header:
                    writer_csv.writeheader()
                writer_csv.writerow(metrics_row)
            json_path = os.path.join(metrics_dir, "metrics.json")
            if os.path.exists(json_path):
                data = json.load(open(json_path, "r"))
            else:
                data = []
            data.append(metrics_row)
            json.dump(data, open(json_path, "w"), indent=2)

            th_path = os.path.join(metrics_dir, "best_threshold.txt")
            with open(th_path, "w") as f:
                f.write(f"{best_thresh:.4f}")
            # 可选：保存每个阈值的指标曲线
            # json.dump(val_metrics_all, open(os.path.join(metrics_dir, "threshold_metrics.json"), "w"), indent=2)

            
            # 检查早停
            if early_stopping(val_loss):
                print(f"\n触发早停,已经{early_stopping.patience}个epoch没有改善")
                break
        
        # 训练结束后快速可视化自检
        if cfg.get("enable_quick_vis", False):
            try:
                vis_dir = os.path.join(run_dir, "quick_vis")
                os.makedirs(vis_dir, exist_ok=True)
                vis_thresholds = eval_thresholds if eval_thresholds else DEFAULT_THRESHOLDS
                samples = cfg.get("quick_vis_samples", 16)
                seed = cfg.get("quick_vis_seed", 42)
                for split_name, ds in [("train", dataset), ("val", val_dataset)]:
                    out_path = os.path.join(vis_dir, f"{split_name}_grid.png")
                    save_quick_visual_grid(
                        lora_sam, ds, device, out_path,
                        thresholds=vis_thresholds,
                        max_samples=samples,
                        seed=seed,
                    )
            except Exception as vis_err:
                print(f"Quick visual check failed: {vis_err}")
        
        # 发送训练完成通知
        lark_send(lark_cb, f"Training done. Best IoU={best_val_iou:.4f}, run_dir={run_dir}")

        # 训练完成
        print("\n训练完成!")
        print(f"最佳验证IoU: {best_val_iou:.4f}")
        print(f"最佳验证F1: {val_metrics['f1']:.4f}")
        print(f"模型已保存至: {best_model_path}")

        # 记录训练耗时
        end_datetime = datetime.now()
        duration_seconds = time.time() - start_time
        duration_str = str(timedelta(seconds=int(duration_seconds)))
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"Run end: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Duration: {duration_seconds:.2f} seconds ({duration_str})\n")
            log_file.write("-" * 40 + "\n\n")
        
        # 关闭TensorBoard写入器
        writer.close()
    except Exception as e:
        import traceback
        print("发生错误:")
        print(traceback.format_exc())
        print(f"错误信息: {str(e)}")
        lark_send(lark_cb, f"Training failed: {e}\nrun_dir={run_dir}")


if __name__ == "__main__":
    main()
