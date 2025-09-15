import torch
import os
import logging
from shutil import copyfile

logger = logging.getLogger(__name__)

class ModelSaver:
    def __init__(self, params):
        self.params = params
        self.best_loss = float('inf')  # 记录当前最优损失

        # 确保保存目录存在
        os.makedirs(self.params.folder_path, exist_ok=True)

    def save_model(self, model, epoch, val_loss):
        """仅在验证损失降低时保存模型"""
        if not self.params.save_model:
            return

        if val_loss < self.best_loss:
            self.best_loss = val_loss  # 更新最优损失
            # save_path = os.path.join(self.params.folder_path, f"{self.params.dataset}_{self.params.model}_best.pth")
            # 根据是否是backdoor模型动态修改保存路径
            suffix = "_backdoor_best.pth" if self.params.is_back_model else "_best.pth"
            save_path = os.path.join(
                self.params.folder_path,
                f"{self.params.dataset}_{self.params.model}{suffix}"
            )
            # 组织需要保存的数据
            saved_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'lr': self.params.lr,
                'params_dict': self.params.to_dict()
            }
            # 保存模型
            torch.save(saved_dict, save_path)
            logger.info(f"New best model saved at epoch {epoch} with val_loss {val_loss:.6f}")
