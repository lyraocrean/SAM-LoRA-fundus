#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from Lora_layers import LoRALayer
from IPython import embed

#  ------------------------------------------------------------------------------------------
#  主要用于在LoRA微调过程中，控制哪些参数是可训练的，并提取与LoRA相关的参数和偏置项的状态字典。
#  通过这种方式，可以有效地减少微调过程中的计算量和内存占用。
#  ------------------------------------------------------------------------------------------

# 标记只有LoRA相关的参数为可训练状态，而其他参数保持不可训练状态。
def mark_only_lora_as_trainable(model):
    """
    冻结模型中除LoRA参数外的所有参数
    """
    # 首先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 然后只解冻LoRA参数
    if hasattr(model, 'lora_As') and hasattr(model, 'lora_Bs'):
        for param in model.lora_As.parameters():
            param.requires_grad = True
        for param in model.lora_Bs.parameters():
            param.requires_grad = True
    
    # 打印可训练参数数量
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"设置了 {len(trainable_params)} 个LoRA参数为可训练状态")

# 提取模型中与LoRA相关的参数和偏置项的状态字典。
def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
