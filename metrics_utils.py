import random
from typing import Iterable, List, Tuple

import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt


DEFAULT_THRESHOLDS: Tuple[float, ...] = (0.5,)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _binarize_gt(mask: torch.Tensor) -> torch.Tensor:
    """Ensure GT is binary {0,1}."""
    mask = mask.float()
    if mask.max() > 1.0:
        mask = (mask > 127).float()
    else:
        mask = (mask > 0.5).float()
    return mask


def _compute_confusion(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    tp = (pred * target).sum(dim=1)
    fp = (pred * (1 - target)).sum(dim=1)
    fn = ((1 - pred) * target).sum(dim=1)
    tn = ((1 - pred) * (1 - target)).sum(dim=1)
    return tp, fp, fn, tn, eps


def _metrics_from_confusion(tp, fp, fn, tn, eps: float = 1e-7):
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2 * precision * recall + eps) / (precision + recall + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "specificity": specificity,
    }


def compute_segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    reduction: str = "mean",
):
    """
    Compute segmentation metrics for a batch.

    Args:
        logits: [B, 1, H, W] raw model outputs.
        targets: [B, 1, H, W] masks in {0,1} or {0,255}.
        thresholds: iterable of thresholds for binarization.
        reduction:
            - "mean": mean over images and thresholds.
            - "per_image": mean over thresholds, per-image values.
            - "per_threshold": mean over images, per-threshold values.
            - "none": return raw tensor per threshold per image.
    Returns:
        metrics: dict of floats or tensors depending on reduction.
        probs: sigmoid probabilities tensor.
    """
    probs = torch.sigmoid(logits)
    targets = _binarize_gt(targets)
    thresholds = tuple(thresholds) if thresholds else DEFAULT_THRESHOLDS

    metric_stack = []
    for t in thresholds:
        preds = (probs > t).float()
        tp, fp, fn, tn, eps = _compute_confusion(preds, targets)
        metric_stack.append(_metrics_from_confusion(tp, fp, fn, tn, eps))

    # Stack: metrics -> [T, B]
    stacked = {k: torch.stack([m[k] for m in metric_stack], dim=0) for k in metric_stack[0]}

    if reduction == "mean":
        return {k: v.mean().item() for k, v in stacked.items()}, probs
    if reduction == "per_image":
        return {k: v.mean(0) for k, v in stacked.items()}, probs
    if reduction == "per_threshold":
        return {k: v.mean(1) for k, v in stacked.items()}, probs
    return stacked, probs


def _denorm(img: torch.Tensor) -> torch.Tensor:
    return (img * _STD.to(img.device) + _MEAN.to(img.device)).clamp(0, 1)


def save_quick_visual_grid(
    model,
    dataset,
    device,
    output_path: str,
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    max_samples: int = 16,
    seed: int = 42,
):
    """
    Sample images from dataset and save a grid: input / GT / pred_prob / pred_mask.
    """
    if len(dataset) == 0:
        return

    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), min(max_samples, len(dataset)))
    images = []
    masks = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, tuple) or isinstance(sample, list):
            img = sample[0]
            mask = sample[1]
        else:
            img = sample["image"]
            mask = sample["mask"]
        images.append(img)
        masks.append(mask)

    imgs_tensor = torch.stack(images).to(device)
    masks_tensor = torch.stack(masks).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(imgs_tensor, multimask_output=False)["masks"]
        metrics, probs = compute_segmentation_metrics(
            outputs, masks_tensor, thresholds=thresholds, reduction="per_image"
        )

    probs = probs.cpu()
    pred_prob = probs[:, 0]
    pred_mask = (pred_prob > thresholds[0]).float()
    masks_show = _binarize_gt(masks_tensor).cpu()[:, 0]

    # Prepare grids
    imgs_show = _denorm(imgs_tensor).cpu()
    input_grid = vutils.make_grid(imgs_show, nrow=4, padding=2)
    gt_grid = vutils.make_grid(masks_show.unsqueeze(1), nrow=4, padding=2)
    prob_grid = vutils.make_grid(pred_prob.unsqueeze(1), nrow=4, padding=2)
    pred_grid = vutils.make_grid(pred_mask.unsqueeze(1), nrow=4, padding=2)

    grids = [
        ("Input", input_grid),
        ("GT", gt_grid),
        ("Pred Prob", prob_grid),
        ("Pred Mask", pred_grid),
    ]

    plt.figure(figsize=(16, 12))
    for i, (title, grid) in enumerate(grids, 1):
        plt.subplot(2, 2, i)
        if grid.shape[0] == 1:
            plt.imshow(grid[0], cmap="gray")
        else:
            plt.imshow(grid.permute(1, 2, 0).numpy())
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return {k: v.mean().item() if torch.is_tensor(v) else v for k, v in metrics.items()}
