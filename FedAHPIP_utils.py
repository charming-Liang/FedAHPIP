"""
Utility functions for FedAHPIP.
"""
import os
import pickle
import numpy as np

from plato.utils import homo_enc


def update_est(config, client_id, data):
    """Update the exposed model weights that can be estimated by adversaries.更新可被对手估计的模型权重"""
    # 提取加密模型中的明文部分
    unencrypted_weights, _, indices = homo_enc.extract_encrypted_model(data)
    # 创建了一个权重向量，将未加密的权重和索引分开处理
    vector_size = len(unencrypted_weights) + len(indices)
    weights_vector = np.zeros(vector_size)

    unencrypted_indices = np.delete(range(vector_size), indices)
    weights_vector[unencrypted_indices] = unencrypted_weights
    # 生成文件路径时，结合了数据源、模型名称和加密比例，可能用于组织存储不同实验条件下的估计结果。
    # 如果启用了随机掩码，路径会加上"_random"后缀。这部分代码在保存估计权重时考虑了不同客户端的ID，确保每个客户端的权重独立存储。
    model_name = config.trainer.model_name
    checkpoint_path = config.params["checkpoint_path"]
    # 生成文件路径 _randoms是随机掩码实验目录，通过目录命名区分不同实验模式，防止数据混淆
    attack_prep_dir = f"{config.data.datasource}_{config.trainer.model_name}_{config.clients.encrypt_ratio}"
    if config.clients.random_mask:
        attack_prep_dir += "_random"
    if not os.path.exists(f"{checkpoint_path}/{attack_prep_dir}/"):
        os.makedirs(f"{checkpoint_path}/{attack_prep_dir}/", exist_ok=True)
    # 保存估计权重
    est_filename = (
        f"{checkpoint_path}/{attack_prep_dir}/{model_name}_est_{client_id}.pth"
    )
    old_est = get_est(est_filename)
    new_est = weights_vector
    # 合并历史估计值
    if old_est is not None:
        weights_vector[indices] = old_est[indices]

    with open(est_filename, "wb") as est_file:
        pickle.dump(new_est, est_file)


def get_est(filename):
    """Load the estimated model, return None if not exists.加载估计的模型权重"""
    # 使用pickle二进制协议保证数据完整性
    try:
        with open(filename, "rb") as est_file:
            return pickle.load(est_file)
    except:
        return None
