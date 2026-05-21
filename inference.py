#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  ============================================================ #
#  Author : Rui Wu
#  Date   : 2025-12-11 
#  Title  : 使用LoRA-SAM进行医学图像分割推理
#  Description: 该脚本加载预训练的LoRA-SAM模型，并在医学图像数据集上进行推理，计算分割指标，并保存结果。
#  ============================================================ #
import sys; sys.stdout.reconfigure(encoding="utf-8")
import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm
import csv
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from prompt_generation import (
    generate_prompts_from_mask,
    generate_box_prompts_from_mask,
    generate_terminal_box_prompts_from_mask,
    encode_boxes_as_sam_points,
)

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 导入SAM相关模块
from segment_anything import sam_model_registry
from add_lora import LoRA_Sam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from metrics_utils import compute_segmentation_metrics, DEFAULT_THRESHOLDS, save_quick_visual_grid

class MedicalDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        is_fives=True,
        use_prompts=False,
        prompt_mode="pos_only",
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
                # 查找对应的mask文件 (文件名完全匹配)
                mask_file = img_file  # FIVES数据集中图像和标签文件名一致
                mask_path = os.path.join(self.mask_dir, mask_file)
                
                if os.path.exists(mask_path):
                    self.valid_pairs.append((img_file, mask_file))
                else:
                    print(f"警告：图像 {img_file} 没有找到对应的mask文件")
        else:
            # 其他数据集结构
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
        
        # 保存原始文件名，用于结果保存
        self.filenames = [img_file for img_file, _ in self.valid_pairs]

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
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
        
        # 应用变换
        if self.transform:
            transformed = self.transform(image=img_np, mask=mask_np)
            img_transformed = transformed['image']
            mask_transformed = transformed['mask']
        else:
            # 如果没有转换，则手动转换为tensor
            img_transformed = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0
            mask_transformed = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0
        
        if self.use_prompts:
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
                "filename": img_file,
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
        
        # 返回原始图像用于可视化
        return img_transformed, mask_transformed, img_file

def collate_with_prompts(batch):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, dict):
        images = torch.stack([b["image"] for b in batch], dim=0)
        masks = torch.stack([b["mask"] for b in batch], dim=0)
        filenames = [b.get("filename") for b in batch]

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
            "filename": filenames,
            "point_coords": point_coords,
            "point_labels": point_labels,
        }

    return default_collate(batch)

def preprocess_images(images, use_clahe=False):
    """
    与训练路径对齐的预处理：
    - 默认直接返回经过 Normalize 的张量
    - 启用 CLAHE 时，先反归一化到 [0,1]，在 CPU 上做增强，再按同一均值/方差归一化
    """
    if not use_clahe:
        return images

    device = images.device
    channels = images.shape[1]
    mean = torch.zeros(1, channels, 1, 1, device=device)
    std = torch.ones(1, channels, 1, 1, device=device)
    if channels == 3:
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # 反归一化到 [0,1] 后在 CPU 上执行 CLAHE
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

def get_transforms(image_size=1024):
    """获取测试数据的转换"""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def visualize_results(
    image,
    true_mask,
    pred_mask,
    output_path,
    positive_points=None,
    negative_points=None,
    box_prompts=None,
):
    """可视化预测结果，并在第三列标出点提示与框提示。"""
    # 将张量转换为NumPy数组
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()
        # 还原归一化
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
    
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()
    
    # 确保掩码是二维的
    if true_mask.ndim > 2:
        true_mask = true_mask.squeeze()
    if pred_mask.ndim > 2:
        pred_mask = pred_mask.squeeze()
    # 将预测转换为二值掩码，便于直观展示
    pred_binary = (pred_mask > 0.5).astype(np.uint8)
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 显示原始图像
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示真实掩码
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('True Mask')
    axes[1].axis('off')
    
    # 显示预测掩码（与输出mask文件一致的二值图）
    axes[2].imshow(pred_binary, cmap='gray')
    axes[2].set_title('Predicted Mask')
    if positive_points:
        xs = [p[0] for p in positive_points]
        ys = [p[1] for p in positive_points]
        axes[2].scatter(xs, ys, c='green', s=20, marker='o', label='positive prompts')
    if negative_points:
        xs = [p[0] for p in negative_points]
        ys = [p[1] for p in negative_points]
        axes[2].scatter(xs, ys, c='red', s=20, marker='x', label='negative prompts')
    if box_prompts:
        for x1, y1, x2, y2 in box_prompts:
            rect = Rectangle(
                (x1, y1),
                max(1.0, x2 - x1),
                max(1.0, y2 - y1),
                linewidth=1.5,
                edgecolor='yellow',
                facecolor='none',
            )
            axes[2].add_patch(rect)
    if positive_points or negative_points:
        axes[2].legend(loc='lower right', fontsize='small')
    elif box_prompts:
        axes[2].legend(
            handles=[Rectangle((0, 0), 1, 1, edgecolor='yellow', facecolor='none', linewidth=1.5, label='box prompts')],
            loc='lower right',
            fontsize='small'
        )
    axes[2].axis('off')
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_mask(mask, output_path):
    """保存分割掩码"""
    # 确保掩码是二维的NumPy数组
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    if mask.ndim > 2:
        mask = mask.squeeze()
    
    # 转换为8位灰度图像
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # 保存图像
    cv2.imwrite(output_path, mask_binary)

def parse_args():
    parser = argparse.ArgumentParser(description='??LoRA-SAM??????????')
    parser.add_argument('--sam_checkpoint', type=str, default='d:\Reasearch\Dataset\sam_vit_b.pth',
                        help='??SAM????????')
    parser.add_argument('--lora_checkpoint', type=str, default='result/runs/2025-12-17-095645-fives/checkpoints/best-iou.pth',
                        help='LoRA????????????')
    parser.add_argument('--test_dir', type=str, default='D:\Reasearch\Dataset\FIVES\FIVES\test',
                        help='????????')
    parser.add_argument('--output_dir', type=str, default='result/runs/inference',
                        help='???????')
    parser.add_argument('--image_size', type=int, default=1024,
                        help='???????')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='??????')
    parser.add_argument('--save_masks', action='store_true', default=True,
                        help='????????')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='?????????')
    parser.add_argument('--is_fives', action='store_true', default=True,
                        help='????FIVES?????')
    parser.add_argument('--thresholds', type=str, default='0.5',
                        help='????????????0.3,0.5,0.7')
    parser.add_argument('--device', type=str, default='cuda',
                        help='?????(cuda ?cpu)')
    parser.add_argument('--threshold_from_file', type=str, default=None,
                    help='文件里只写一个阈值，优先使用它')
    parser.add_argument('--use_prompts', action=argparse.BooleanOptionalAction, default=True,
                        help='是否在推理时使用GT mask自动生成点提示（默认开启）')
    parser.add_argument('--prompt_mode', type=str, choices=['pos_only', 'pos_neg', 'box_only', 'box_terminal'], default='pos_only',
                        help='prompt类型：仅正点(pos_only)、正负点(pos_neg)、小框(box_only)或末梢优先小框(box_terminal)')
    parser.add_argument('--prompt_edge_distance', type=int, default=4,
                        help='正点距边界上限，与训练/验证一致')
    parser.add_argument('--prompt_neg_min_distance', type=int, default=5,
                        help='供生成函数维持一致性的负点距离参数（pos_neg模式下会生成负点）')
    parser.add_argument('--prompt_box_min_count', type=int, default=3,
                        help='box_only模式下每张图最少框数')
    parser.add_argument('--prompt_box_max_count', type=int, default=4,
                        help='box_only模式下每张图最多框数')
    parser.add_argument('--prompt_box_size', type=int, default=96,
                        help='box_only模式下小框边长（像素）')
    parser.add_argument('--prompt_box_min_component_area', type=int, default=50,
                        help='box_only模式下参与采样的最小连通域面积')
    parser.add_argument('--prompt_box_min_center_distance', type=int, default=40,
                        help='box_only模式下框中心最小间距（像素）')
    parser.add_argument('--prompt_box_terminal_radius', type=int, default=8,
                        help='box_terminal模式下端点邻域半径（像素）')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    if args.threshold_from_file and os.path.exists(args.threshold_from_file):
        with open(args.threshold_from_file) as f:
            thresholds = (float(f.read().strip()),)
    else:
        thresholds = tuple(float(t) for t in args.thresholds.split(",") if t.strip()) or DEFAULT_THRESHOLDS
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)
    
    # 选择设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载原始SAM模型
    print("正在加载SAM基础模型...")
    sam = sam_model_registry["vit_b"](checkpoint=args.sam_checkpoint)
    sam.to(device)
    
    # 创建LoRA模型实例
    print("正在初始化LoRA模型...")
    lora_sam = LoRA_Sam(sam, 4, lora_alpha=9, lora_dropout=0.1)
    lora_sam.to(device)
    
    # 加载训练好的LoRA权重
    print(f"正在加载LoRA权重: {args.lora_checkpoint}")
    checkpoint = torch.load(args.lora_checkpoint, map_location=device)
    
    # 处理键名不匹配问题
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # 创建新的状态字典
        new_state_dict = {}
        for k, v in state_dict.items():
            # 将sam_model前缀替换为空，因为LoRA_Sam使用sam而不是sam_model
            if k.startswith('sam_model.'):
                new_key = k.replace('sam_model.', 'sam.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        # 尝试加载新的状态字典
        try:
            missing_keys, unexpected_keys = lora_sam.load_state_dict(new_state_dict, strict=False)
            print(f"加载状态字典 - 缺失键: {missing_keys}, 意外键: {unexpected_keys}")
            expected_keys = set([k for k in lora_sam.state_dict().keys()])
            loaded_keys = set([k for k in new_state_dict.keys()])
            missing = expected_keys - loaded_keys
            print(f"缺失的参数键: {missing}")
            if len(missing) > 0:
                print("警告: 模型加载不完整，可能影响性能")
        except Exception as e:
            print(f"加载状态字典时出错: {e}")
            print("尝试使用compatibility=True选项进行加载...")
            # 如果有需要可以添加额外的兼容性处理
    else:
        print("警告：模型检查点中未找到'model_state_dict'键")
        lora_sam.load_state_dict(checkpoint, strict=False)
        
    print(f"加载完成! 训练epoch: {checkpoint.get('epoch', 'N/A')}, 验证IoU: {checkpoint.get('val_iou', 'N/A')}")
    
    # 切换到评估模式
    lora_sam.eval()
    
    # 准备测试数据集
    print("正在准备测试数据...")
    test_transform = get_transforms(image_size=args.image_size)
    test_dataset = MedicalDataset(
        args.test_dir,
        transform=test_transform,
        is_fives=args.is_fives,
        use_prompts=args.use_prompts,
        prompt_mode=args.prompt_mode,
        prompt_edge_distance=args.prompt_edge_distance,
        prompt_neg_min_distance=args.prompt_neg_min_distance,
        prompt_box_min_count=args.prompt_box_min_count,
        prompt_box_max_count=args.prompt_box_max_count,
        prompt_box_size=args.prompt_box_size,
        prompt_box_min_component_area=args.prompt_box_min_component_area,
        prompt_box_min_center_distance=args.prompt_box_min_center_distance,
        prompt_box_terminal_radius=args.prompt_box_terminal_radius,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_with_prompts,
    )
    
    # Collect metrics
    metric_keys = ['iou', 'dice', 'precision', 'recall', 'f1', 'accuracy', 'specificity']
    all_metrics = {k: 0.0 for k in metric_keys}
    sample_metrics = []
    total_samples = 0
    
    # Run inference
    print("Start inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Inference Process", ascii=True, ncols=80)):
            if isinstance(batch, dict):
                images = batch["image"]
                masks = batch["mask"]
                filenames = batch.get("filename", [])
                point_coords = batch.get("point_coords")
                point_labels = batch.get("point_labels")
            else:
                images, masks, filenames = batch
                point_coords = None
                point_labels = None

            if point_coords is not None and point_coords.numel() == 0:
                point_coords = None
            if point_labels is not None and point_labels.numel() == 0:
                point_labels = None

            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            if point_coords is not None:
                point_coords = point_coords.to(device)
            if point_labels is not None:
                point_labels = point_labels.to(device)

            points = (point_coords, point_labels) if (point_coords is not None and point_labels is not None) else None
            
            # Ensure mask shape is correct
            if masks.dim() == 3:
                masks = masks.unsqueeze(1)
            
            # Preprocess images
            processed_images = preprocess_images(images, use_clahe=False)
            
            # Forward pass
            outputs = lora_sam(processed_images, points=points, multimask_output=False)
            pred_masks = outputs["masks"]
            
            # Resize prediction to target mask size if needed
            if pred_masks.shape[-2:] != masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
            
            batch_metrics, probs = compute_segmentation_metrics(
                pred_masks,
                masks,
                thresholds=thresholds,
                reduction="per_image",
            )
            pred_probs = probs[:, 0]
            
            # Per-sample metrics and visualization
            for i in range(images.size(0)):
                if batch_idx * args.batch_size + i >= len(test_dataset.filenames):
                    break
                
                filename = filenames[i]
                pred_prob = pred_probs[i]
                true_mask = masks[i, 0]
                
                metrics_avg = {k: batch_metrics[k][i].item() for k in metric_keys}
                
                # Accumulate sample metrics
                sample_metric = {
                    'filename': filename,
                    **metrics_avg
                }
                sample_metrics.append(sample_metric)
                total_samples += 1
                
                # Accumulate totals
                for k in metric_keys:
                    all_metrics[k] += metrics_avg[k]
                
                # Save binary mask & probability map
                if args.save_masks:
                    mask_path = os.path.join(args.output_dir, 'masks', filename)
                    save_mask(pred_prob, mask_path)
                    prob_dir = os.path.join(args.output_dir, 'prob_maps')
                    os.makedirs(prob_dir, exist_ok=True)
                    prob_path = os.path.join(prob_dir, filename)
                    prob_img = (pred_prob.cpu().numpy() * 255).astype(np.uint8)
                    cv2.imwrite(prob_path, prob_img)
                
                # Visualization with prompt points
                if args.visualize:
                    vis_path = os.path.join(args.output_dir, 'visualizations', os.path.splitext(filename)[0] + '.png')
                    orig_image = images[i].cpu()
                    pos_points = None
                    neg_points = None
                    box_prompts = None
                    if point_coords is not None and point_labels is not None and i < point_coords.shape[0]:
                        coords_i = point_coords[i].detach().cpu()
                        labels_i = point_labels[i].detach().cpu()
                        pos_mask = labels_i == 1
                        neg_mask = labels_i == 0
                        if pos_mask.any():
                            coords_xy = coords_i[pos_mask]
                            pos_points = [(float(x), float(y)) for x, y in coords_xy]
                        if neg_mask.any():
                            coords_xy = coords_i[neg_mask]
                            neg_points = [(float(x), float(y)) for x, y in coords_xy]
                        # Decode SAM box prompts from corner labels: 2=top-left, 3=bottom-right
                        valid_mask = labels_i >= 0
                        coords_valid = coords_i[valid_mask]
                        labels_valid = labels_i[valid_mask]
                        decoded_boxes = []
                        n = labels_valid.shape[0]
                        j = 0
                        while j + 1 < n:
                            l0 = int(labels_valid[j].item())
                            l1 = int(labels_valid[j + 1].item())
                            if l0 == 2 and l1 == 3:
                                x1, y1 = coords_valid[j].tolist()
                                x2, y2 = coords_valid[j + 1].tolist()
                                x_min, x_max = sorted([float(x1), float(x2)])
                                y_min, y_max = sorted([float(y1), float(y2)])
                                decoded_boxes.append((x_min, y_min, x_max, y_max))
                                j += 2
                            else:
                                j += 1
                        if decoded_boxes:
                            box_prompts = decoded_boxes
                    visualize_results(
                        orig_image,
                        true_mask,
                        pred_prob,
                        vis_path,
                        positive_points=pos_points,
                        negative_points=neg_points,
                        box_prompts=box_prompts,
                    )
    
    # Compute average metrics
    num_samples = max(total_samples, 1)
    avg_metrics = {k: v / num_samples for k, v in all_metrics.items()}
    
    # 打印平均指标
    print("\n评估指标:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 保存样本指标到CSV
    csv_path = os.path.join(args.output_dir, 'sample_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'iou', 'dice', 'precision', 'recall', 'f1', 'accuracy', 'specificity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in sample_metrics:
            writer.writerow(metric)
    
    # 保存平均指标到CSV
    avg_csv_path = os.path.join(args.output_dir, 'average_metrics.csv')
    with open(avg_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for metric, value in avg_metrics.items():
            writer.writerow([metric, f"{value:.6f}"])
    
    # 保存指标到pickle文件，与训练结果格式兼容
    metrics_dict = {
        'test_' + k: v for k, v in avg_metrics.items()
    }
    with open(os.path.join(args.output_dir, 'test_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_dict, f)
    
    print(f"\n推理完成! 结果已保存至: {args.output_dir}")

if __name__ == "__main__":
    main() 
