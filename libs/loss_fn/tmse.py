import torch
import torch.nn as nn
import torch.nn.functional as F


class TMSE(nn.Module):
    """
    Temporal MSE Loss Function
    Proposed in Y. A. Farha et al. MS-TCN: Multi-Stage Temporal Convolutional Network for ActionSegmentation in CVPR2019
    arXiv: https://arxiv.org/pdf/1903.01945.pdf
    """

    def __init__(self, threshold: float = 4, ignore_index: int = 255) -> None:
        super().__init__()
        self.threshold = threshold  # 平滑最大差阈值=4
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, preds: torch.Tensor, gts: torch.Tensor) -> torch.Tensor:

        total_loss = 0.0
        batch_size = preds.shape[0]
        for pred, gt in zip(preds, gts): # 遍历每个样本的预测值和标签
            pred = pred[:, torch.where(gt != self.ignore_index)[0]] # 根据 ignore_index 剔除标签中的指定索引位置

            loss = self.mse( # 计算 TMSE 损失，包括 log_softmax 操作、MSE 损失计算、裁剪等步骤
                F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            )

            loss = torch.clamp(loss, min=0, max=self.threshold ** 2) # 对损失进行裁剪
            total_loss += torch.mean(loss)# 将每个样本的损失加总

        return total_loss / batch_size


class GaussianSimilarityTMSE(nn.Module):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(
        self, threshold: float = 4, sigma: float = 1.0, ignore_index: int = 255
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma # Gaussian Similarity 权重计算中的高斯分布标准差

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]

        b, c, v, t = sim_index.size() #模型的输入
        sim_index = sim_index.reshape(b, c*v, t)

        for pred, gt, sim in zip(preds, gts, sim_index): # 遍历每个样本的预测值、标签和相似性索引,提取非填充的有效部分
            pred = pred[:, torch.where(gt != self.ignore_index)[0]] # 根据 ignore_index 剔除标签中的指定索引位置
            sim = sim[:, torch.where(gt != self.ignore_index)[0]] # 同样剔除相似性索引中的指定索引位置

            # calculate gaussian similarity
            diff = sim[:, 1:] - sim[:, :-1] # 计算高斯相似性权重
            similarity = torch.exp(-torch.norm(diff, dim=0) / (2 * self.sigma ** 2)) #norm为范数，就是C通道的模 （C，T-1）->（T-1）

            # calculate temporal mse
            loss = self.mse(  # 计算时间均方误差
                F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred[:, :-1], dim=1)
            )
            loss = torch.clamp(loss, min=0, max=self.threshold ** 2)
            # gaussian similarity weighting
            loss = similarity * loss # 进行高斯相似性权重加权

            total_loss += torch.mean(loss)

        return total_loss / batch_size  # 返回平均损失
