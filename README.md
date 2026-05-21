# SAM + LoRA for Retinal Vessel Segmentation

Parameter-efficient fine-tuning of Meta's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything) for retinal fundus vessel segmentation using Low-Rank Adaptation (LoRA).

SAM is a powerful general-purpose segmentation foundation model, but it is designed for natural images and performs poorly on medical images out of the box (Dice = 0.110 on retinal vessels). This project injects LoRA adapters into SAM's image encoder, training only a small number of additional parameters while keeping the base model frozen — achieving Dice = 0.849 on the FIVES dataset.

---

## Key Results (FIVES Test Set)

| Method | Prompt | Dice | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| SAM (zero-shot) | pos + neg points | 0.110 | 0.171 | 0.305 | 0.739 |
| **SAM + LoRA** | **none (prompt-free)** | **0.847** | **0.835** | **0.869** | **0.981** |
| SAM + LoRA (no-prompt train) | pos point | 0.805 | 0.733 | 0.901 | 0.972 |
| SAM + LoRA (pos+neg train) | pos point | 0.847 | 0.840 | 0.868 | 0.981 |
| SAM + LoRA (pos+neg train) | pos + neg points | 0.849 | 0.841 | 0.866 | 0.981 |

The **prompt-free** variant (row 2) is the most practically useful: it requires no user interaction at inference time, making it directly deployable as an automatic segmentation pipeline.

---

## Method

### LoRA Injection

LoRA adapters are injected into the **QKV projection layers** of every attention block in SAM's ViT-B image encoder. For each frozen weight matrix W, we add a low-rank residual:

```
output = W·x + (α/r) · B·A·x
```

where A ∈ ℝ^(r×d) is initialized with small random values and B ∈ ℝ^(d×r) is zero-initialized, so the adapter contributes nothing at the start of training. Only A and B are updated; all original SAM weights remain frozen.

**Hyperparameters:** rank r = 4, scale α = 9, dropout = 0.1.

Optionally, the mask decoder can also be unfrozen for additional fine-tuning.

### Prompt-Free Inference

Standard SAM requires user-provided point or box prompts to produce a segmentation. During training, we optionally supply positive/negative point prompts derived from the ground-truth mask. The key finding is that after LoRA adaptation, the model learns to encode vessel-specific features in the image embedding directly — so at inference time the prompt encoder can be bypassed entirely, yielding competitive results without any user input.

### Loss Function

Training uses a composite loss:

```
L = λ₁ · BCE + λ₂ · Dice + λ₃ · IoU_score
```

with focal weighting on the BCE term to handle severe class imbalance (vessels occupy < 5% of fundus image pixels).

### Training Details

| Setting | Value |
|---|---|
| Backbone | SAM ViT-B |
| LoRA rank / alpha | 4 / 9 |
| Optimizer | AdamW |
| Learning rate | 1×10⁻⁴ |
| LR schedule | CosineAnnealingWarmRestarts |
| Batch size | 2 (×2 accumulation steps) |
| Image size | 1024×1024 |
| Mixed precision | Yes (torch.amp) |
| Early stopping | Yes (patience = 7) |
| Dataset | FIVES (train/test split) |

---

## Dataset

This project uses the **FIVES** dataset (Fundus Image VEssel Segmentation):

> Jin, K., Huang, X., Zhou, J., et al. "FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation." *Scientific Data*, 2022.

FIVES contains 800 high-resolution (2048×2048) color fundus images with pixel-level vessel annotations across four disease categories (normal, AMD, DR, glaucoma).

**The dataset is not included in this repository.** Download from the [official source](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169/1) and organize as:

```
data/
  FIVES/
    train/
      Original/   ← .png images
      Ground truth/  ← .png masks (same filename)
    test/
      Original/
      Ground truth/
```

---

## Repository Structure

```
sam-lora-fundus/
├── train_lora_sam.py     # Training script (CONFIG + CLI args)
├── inference.py          # Inference + evaluation script
├── add_lora.py           # LoRA injection into SAM image encoder
├── Lora_layers.py        # LoRA layer implementations (Linear, ConvLoRA)
├── Lora_utils.py         # LoRA weight save/load utilities
├── metrics_utils.py      # Segmentation metrics (Dice, IoU, etc.)
├── prompt_generation.py  # Point and box prompt generation from masks
└── sam_test/
    └── sam_only_results/ # Zero-shot SAM baseline results
```

---

## Setup

**1. Install SAM:**

```bash
pip install git+https://github.com/facebookresearch/segment-anything.git
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Download SAM ViT-B checkpoint:**

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Training

```bash
python train_lora_sam.py \
  --checkpoint_path /path/to/sam_vit_b_01ec64.pth \
  --data_path /path/to/FIVES/train \
  --test_path /path/to/FIVES/test
```

Key CLI arguments:

| Argument | Default | Description |
|---|---|---|
| `--checkpoint_path` | required | Path to `sam_vit_b_01ec64.pth` |
| `--data_path` | required | Path to training data directory |
| `--test_path` | required | Path to test data directory |
| `--num_epochs` | 100 | Maximum training epochs |
| `--batch_size` | 2 | Batch size per GPU |
| `--learning_rate` | 1e-4 | Initial learning rate |
| `--use_prompts` | True | Enable point prompts during training |
| `--prompt_mode` | `pos_neg` | `pos_neg` or `pos_only` |
| `--resume` | None | Path to checkpoint to resume from |
| `--run_name` | `fives` | Name for this run (used in output paths) |

Notifications via Lark (Feishu) webhook are supported through environment variables `LARK_WEBHOOK` and `LARK_SECRET`.

---

## Inference

```bash
python inference.py \
  --checkpoint /path/to/saved_lora.pth \
  --sam_checkpoint /path/to/sam_vit_b_01ec64.pth \
  --data_path /path/to/FIVES/test \
  --prompt_free
```

---

## License

This project is released under the [MIT License](LICENSE).

It builds on Meta's [Segment Anything Model (SAM)](https://github.com/facebookresearch/segment-anything), which is released under the Apache 2.0 License.

---

## Citation

If you use this code, please cite the original SAM paper:

```bibtex
@article{kirillov2023sam,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and
          Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and
          Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
```

and the FIVES dataset:

```bibtex
@article{jin2022fives,
  title={FIVES: A Fundus Image Dataset for AI-based Vessel Segmentation},
  author={Jin, Kai and Huang, Xingru and Zhou, Jingxing and Li, Yunxiang and
          Yan, Yan and Sun, Yibao and Zhang, Qianni and Wang, Yaqi and Ye, Juan},
  journal={Scientific Data},
  volume={9},
  number={1},
  pages={475},
  year={2022}
}
```
