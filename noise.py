import numpy as np
import torch

class GaussianNoise:
    def __init__(self, 
                 epsilon: float,
                 delta: float,
                 epoch: int,
                 clip: float,
                 lr: float,
                 train_len: int,
                 device: str = 'cpu'
                ):
        self.epsilon = epsilon
        self.delta = delta
        self.epoch = epoch
        self.clip = clip
        self.lr = lr
        self.train_len = train_len
        self.device = device

        # 计算单次查询的隐私参数
        self.epsilon_single_query = self.epsilon / epoch
        self.delta_single_query = self.delta / epoch
        
        # 计算噪声尺度
        self.noise_scale = np.sqrt(2 * np.log(1.25 / self.delta_single_query)) / self.epsilon_single_query

    def clip_gradients(self, model):
        """使用L2范数进行梯度裁剪"""
        self._per_sample_clip(model, norm=2)

    def add_noise(self, model):
        """添加高斯噪声"""
        sensitivity = self._cal_sensitivity()
        state_dict = model.state_dict()
        
        with torch.no_grad():
            for k, v in state_dict.items():
                noise = torch.from_numpy(
                    np.random.normal(
                        loc=0,
                        scale=sensitivity * self.noise_scale,
                        size=v.shape
                    )
                ).to(self.device)
                state_dict[k] += noise
        model.load_state_dict(state_dict)

    def _per_sample_clip(self, model, norm=2):
        """逐样本梯度裁剪"""
        for _, param in model.named_parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data / max(1, param.grad.data.norm(norm) / self.clip)

    def _cal_sensitivity(self) -> float:
        """计算敏感度"""
        return 2 * self.lr * self.clip / self.train_len