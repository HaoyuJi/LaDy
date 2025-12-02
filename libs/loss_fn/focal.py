from typing import Optional

import torch
import torch.nn as nn


class FocalLoss(nn.Module): #Focal Loss 的定义基于交叉熵损失，并对其进行了修改以关注难以分类的样本。
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average: bool = True,
        batch_average: bool = True,
        ignore_index: int = 255,
        gamma: float = 2.0,
        alpha: float = 0.25,
    ) -> None:
        super().__init__()

        self.gamma = gamma # Focal Loss 中的超参数 gamma
        self.alpha = alpha # Focal Loss 中的超参数 alpha
        self.batch_average = batch_average # 是否进行 batch 平均
        self.criterion = nn.CrossEntropyLoss(  # 使用交叉熵损失作为 Focal Loss 的基础损失函数
            weight=weight, ignore_index=ignore_index, size_average=size_average
        )

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n, _, _ = logit.size() # 获取输入 logit 的维度信息

        logpt = -self.criterion(logit, target.long()) # 计算交叉熵损失的 log(exp(-x)) 部分
        pt = torch.exp(logpt)

        if self.alpha is not None: # 如果指定了 alpha 参数，则将计算得到的 logpt 乘以 alpha
            logpt *= self.alpha

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.batch_average: # 如果指定了 batch_average 参数，则进行 batch 平均
            loss /= n

        return loss #Focal Loss 的定义基于交叉熵损失，并对其进行了修改以关注难以分类的样本。
