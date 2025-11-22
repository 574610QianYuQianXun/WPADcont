import os
import torch

class ModelSaver:
    def __init__(self, params):
        self.params = params
        self.best_acc = 0.0  # 当前最优准确率
        os.makedirs(self.params.folder_path, exist_ok=True)

    def save_checkpoint(self, model, epoch, val_loss, val_accuracy, back_loss=None, back_accuracy=None):
        """根据准确率保存最优模型：
           - 干净模型使用 val_accuracy；
           - 后门模型使用 back_accuracy。
        """
        if not self.params.save_model:
            return

        # === 选择用于判断的准确率 ===
        if self.params.is_back_model:
            acc_to_compare = back_accuracy
            acc_type = "backdoor"
        else:
            acc_to_compare = val_accuracy
            acc_type = "clean"

        # === 跳过无效值 ===
        if acc_to_compare is None:
            print(f"⚠️ Warning: {acc_type} accuracy is None, skip saving.")
            return

        # === 判断是否最佳模型 ===
        if acc_to_compare > self.best_acc:
            self.best_acc = acc_to_compare

            # ----★ 新增部分：攻击方法名称处理 ★----
            attack_suffix = ""
            if self.params.attack_type != "How_backdoor":
                attack_suffix = f"_{self.params.attack_type}"

            model_suffix = "_backdoor_best.pth" if self.params.is_back_model else "_best.pth"

            save_path = os.path.join(
                self.params.folder_path,
                f"{self.params.dataset}_{self.params.model}{attack_suffix}{model_suffix}"
            )

            saved_dict = {
                'state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'back_loss': back_loss,
                'back_accuracy': back_accuracy,
                'lr': self.params.lr,
                'params_dict': self.params.to_dict()
            }

            torch.save(saved_dict, save_path)
            print(f"✅ New best {acc_type} model saved at epoch {epoch} with acc {acc_to_compare:.4f}")

