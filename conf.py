# Dataset Non-iid configs
seed=2024
dir_alpha=0.5

# Prune Configs
max_sp_rounds=2     # 最大结构化剪枝次数
prune_threshold=0.5
prune_step=2
start_ratio=0.2
end_ratio=0.4 ### 修改为client info prune ratio max
channel_sparsity=0.2 ### Temp 0.2 0.64|0.5 0.8 |0.3 
min_inscrease=0.05

# Learning Rate Configs
#lr=0.02
lr=0.02
min_lr=0.001
decrease_rate=0.1
decrease_frequency=2    # epoch
decreasing_lr=80,120
