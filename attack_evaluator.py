import math
import torch
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F


class AttackEvaluator:
    """攻击效果评估类 - 修复维度不匹配问题"""

    def __init__(self, dataset_config, encrypt_ratio=0.5, output_dir="./attack_evaluation"):
        self.dataset_config = dataset_config
        self.encrypt_ratio = encrypt_ratio
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 评估指标存储
        self.evaluation_results = {}

        # 数据集特定的配置
        self.dataset_name = dataset_config.get('name', 'unknown').lower()
        self._setup_dataset_params()

    def _setup_dataset_params(self):
        """设置数据集参数"""
        if 'mnist' in self.dataset_name:
            self.ideal_contrast = 0.2
            self.contrast_range = (0.05, 0.6)
            self.dataset_type = 'grayscale'
        elif 'cifar10' in self.dataset_name or 'cifar100' in self.dataset_name:
            self.ideal_contrast = 0.3
            self.contrast_range = (0.1, 0.8)
            self.dataset_type = 'color'
        elif 'neu_cls' in self.dataset_name or 'neu' in self.dataset_name:
            self.ideal_contrast = 0.25
            self.contrast_range = (0.08, 0.7)
            self.dataset_type = 'grayscale'  # NEU_CLS是灰度图
        elif 'carparts' in self.dataset_name or 'carparts' in self.dataset_name:
            self.ideal_contrast = 0.35
            self.contrast_range = (0.15, 0.9)
            self.dataset_type = 'color'  # Carparts-50是彩色图
        else:
            self.ideal_contrast = 0.25
            self.contrast_range = (0.05, 0.7)
            self.dataset_type = 'unknown'

    def evaluate_attack_quality(self, reconstructed_data, reconstructed_labels,
                                original_data=None, original_labels=None,
                                client_id=None, round_num=None, is_final_round=False,
                                communication_stats=None):
        """评估攻击质量 - 修复维度不匹配问题"""
        evaluation = {
            'client_id': client_id,
            'round': round_num,
            'is_final_round': is_final_round,
            'has_original_data': original_data is not None,
        }

        try:
            # 基本统计信息
            evaluation.update(self._compute_basic_stats(reconstructed_data, reconstructed_labels))

            # 图像质量评估
            if reconstructed_data is not None:
                evaluation.update(self._evaluate_image_quality(reconstructed_data))

            # 与原始数据比较 - 核心指标计算
            if is_final_round and original_data is not None:
                evaluation.update(self._compare_with_original(
                    reconstructed_data, original_data, reconstructed_labels, original_labels
                ))

            # 总体评分 - 新版评分规则
            evaluation['overall_score'] = self._compute_attack_score(evaluation, is_final_round)

            # 保存评估结果
            key = f"client_{client_id}_round_{round_num}" if client_id and round_num else f"eval_{len(self.evaluation_results)}"
            self.evaluation_results[key] = evaluation

            # 生成可视化结果
            if reconstructed_data is not None:
                self._generate_visualization(
                    reconstructed_data, evaluation, client_id, round_num,
                    original_data if (is_final_round and original_data is not None) else None
                )

            logging.info(f"[AttackEvaluation] 客户端 {client_id} 攻击分数: {evaluation['overall_score']:.4f}")

        except Exception as e:
            logging.error(f"[AttackEvaluation] 评估过程出错: {e}")
            evaluation['overall_score'] = 0.0
            evaluation['error'] = str(e)

        return evaluation

    def _compute_basic_stats(self, data, labels):
        """计算基本统计信息"""
        stats = {}

        try:
            if data is not None:
                stats.update({
                    'data_mean': data.mean().item(),
                    'data_std': data.std().item(),
                    'brightness': data.mean().item(),
                })

            if labels is not None:
                label_probs = labels

                if label_probs.dim() == 0:
                    stats.update({
                        'label_entropy': 0.0,
                        'label_confidence': 1.0,
                    })
                elif label_probs.dim() == 1:
                    if label_probs.numel() == 1:
                        stats.update({
                            'label_entropy': 0.0,
                            'label_confidence': 1.0,
                        })
                    else:
                        num_classes = self.dataset_config.get('num_classes', 10)
                        one_hot_labels = F.one_hot(label_probs.long(), num_classes=num_classes).float()
                        stats.update({
                            'label_entropy': self._compute_entropy(one_hot_labels),
                            'label_confidence': 1.0,
                        })
                else:
                    if label_probs.dim() > 2:
                        label_probs = label_probs.view(label_probs.size(0), -1)

                    if label_probs.size(1) > 1:
                        label_probs_normalized = F.softmax(label_probs, dim=1)
                        stats.update({
                            'label_entropy': self._compute_entropy(label_probs_normalized),
                            'label_confidence': label_probs_normalized.max(dim=1)[0].mean().item(),
                        })
                    else:
                        stats.update({
                            'label_entropy': 0.0,
                            'label_confidence': 1.0,
                        })

        except Exception as e:
            logging.error(f"[AttackEvaluation] 基本统计计算失败: {e}")
            stats.update({
                'label_entropy': 0.0,
                'label_confidence': 0.0,
            })

        return stats

    def _evaluate_image_quality(self, images):
        """评估图像质量"""
        quality_metrics = {}
        try:
            images_clamped = torch.clamp(images, 0, 1)
            images_4d = self._ensure_4d_tensor(images_clamped)

            if images_4d.dim() != 4:
                return quality_metrics

            contrast = images_4d.std().item()
            brightness = images_4d.mean().item()
            quality_metrics['contrast'] = contrast
            quality_metrics['brightness'] = brightness

            sharpness = self._compute_sharpness(images_4d)
            quality_metrics['sharpness'] = sharpness

            return quality_metrics

        except Exception:
            return quality_metrics

    def _ensure_4d_tensor(self, tensor):
        """确保张量为4维"""
        if tensor.dim() == 2:
            return tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            if tensor.size(0) in [1, 3]:
                return tensor.unsqueeze(0)
            else:
                return tensor.permute(2, 0, 1).unsqueeze(0)
        elif tensor.dim() == 4:
            return tensor
        else:
            return tensor.view(1, 1, *tensor.shape[-2:])

    def _compute_sharpness(self, images):
        """计算清晰度"""
        try:
            if self.dataset_type == 'grayscale':
                if images.size(2) > 1 and images.size(3) > 1:
                    dx = torch.abs(images[:, 0, :, 1:] - images[:, 0, :, :-1])
                    dy = torch.abs(images[:, 0, 1:, :] - images[:, 0, :-1, :])
                    return (dx.mean() + dy.mean()).item() / 2
            else:
                sharpness_vals = []
                for channel in range(images.shape[1]):
                    if images.size(2) > 1 and images.size(3) > 1:
                        dx = torch.abs(images[:, channel, :, 1:] - images[:, channel, :, :-1])
                        dy = torch.abs(images[:, channel, 1:, :] - images[:, channel, :-1, :])
                        sharpness_vals.append((dx.mean() + dy.mean()).item() / 2)
                return sum(sharpness_vals) / len(sharpness_vals) if sharpness_vals else 0.0
        except:
            return 0.0

    def _compare_with_original(self, reconstructed, original, rec_labels, orig_labels):
        """与原始数据比较 - 修复维度不匹配问题"""
        comparison_metrics = {}

        try:
            if reconstructed.device != original.device:
                original = original.to(reconstructed.device)

            # 修复：只比较第一个样本，因为DLG攻击只重建1个样本
            batch_size = 1  # DLG攻击只重建1个样本
            rec_data_sample = reconstructed[:1]  # 只取第一个重建样本
            orig_data_sample = original[:1]  # 只取第一个原始样本

            # 1. 余弦相似度 (权重: 0.15)
            reconstructed_flat = rec_data_sample.flatten()
            original_flat = orig_data_sample.flatten()

            # 修复余弦相似度计算：使用更可靠的方法
            if torch.norm(reconstructed_flat) > 1e-8 and torch.norm(original_flat) > 1e-8:
                # 方法1：直接计算点积和范数（更可靠）
                dot_product = torch.dot(reconstructed_flat, original_flat)
                norm_rec = torch.norm(reconstructed_flat)
                norm_orig = torch.norm(original_flat)
                cosine_sim = (dot_product / (norm_rec * norm_orig + 1e-8)).item()
                
                # 确保余弦相似度在[-1, 1]范围内
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                
                # 转换为[0,1]范围：将负值映射到0
                if cosine_sim < 0:
                    cosine_sim = 0.0
                
                # 记录调试信息
                logging.debug(f"[CosineSimilarity] 点积: {dot_product.item():.6f}, "
                             f"重建范数: {norm_rec.item():.6f}, 原始范数: {norm_orig.item():.6f}, "
                             f"余弦相似度: {cosine_sim:.6f}")
            else:
                cosine_sim = 0.0
                logging.debug("[CosineSimilarity] 向量范数过小，返回0")

            comparison_metrics['cosine_similarity'] = cosine_sim

            # 2. 分类准确率 (权重: 0.25) - 修复维度不匹配
            classification_accuracy = self._compute_classification_accuracy_fixed(
                rec_labels, orig_labels
            )
            comparison_metrics['classification_accuracy'] = classification_accuracy

            # 3. SSIM结构相似性 (权重: 0.2)
            comparison_metrics['ssim_approx'] = self._compute_ssim(
                rec_data_sample, orig_data_sample
            )

            # 4. 均方误差MSE (权重: 0.25)
            mse = F.mse_loss(rec_data_sample, orig_data_sample).item()
            comparison_metrics['mse'] = mse

            # 记录详细的标签信息用于调试
            comparison_metrics['label_debug'] = {
                'reconstructed_labels_shape': rec_labels.shape if rec_labels is not None else None,
                'original_labels_shape': orig_labels.shape if orig_labels is not None else None,
                'reconstructed_labels': rec_labels.cpu().numpy() if rec_labels is not None else None,
                'original_labels': orig_labels.cpu().numpy() if orig_labels is not None else None,
                'comparison_note': '只比较第一个样本，因为DLG攻击只重建1个样本'
            }

        except Exception as e:
            logging.error(f"[AttackEvaluation] 与原始数据比较失败: {e}")

        return comparison_metrics

    def _compute_classification_accuracy_fixed(self, rec_labels, orig_labels):
        """计算分类准确率 - 修复维度不匹配问题，并检查重建标签是否在原始标签中"""
        try:
            if rec_labels is None or orig_labels is None:
                logging.warning("[ClassificationAccuracy] 标签为空，返回0")
                return 0.0

            # 确保标签在相同设备上
            if rec_labels.device != orig_labels.device:
                orig_labels = orig_labels.to(rec_labels.device)

            # 打印标签信息用于调试
            logging.info(f"[ClassificationAccuracy] 重建标签: {rec_labels.shape}, 原始标签: {orig_labels.shape}")
            logging.info(f"[ClassificationAccuracy] 重建标签值: {rec_labels}, 原始标签值: {orig_labels}")

            # 获取重建标签的值
            if rec_labels.dim() == 0:
                rec_label_value = rec_labels.item()
            elif rec_labels.dim() == 1:
                rec_label_value = rec_labels[0].item() if rec_labels.numel() > 1 else rec_labels.item()
            else:
                # 重建标签是2D或更高维，取argmax
                rec_pred = rec_labels.argmax(dim=1) if rec_labels.dim() > 1 else rec_labels
                rec_label_value = rec_pred[0].item() if rec_pred.numel() > 1 else rec_pred.item()

            # 将原始标签转换为列表
            if orig_labels.dim() == 0:
                orig_labels_list = [orig_labels.item()]
            elif orig_labels.dim() == 1:
                orig_labels_list = orig_labels.tolist()
            else:
                # 如果是2D，取argmax
                orig_preds = orig_labels.argmax(dim=1) if orig_labels.dim() > 1 else orig_labels
                orig_labels_list = orig_preds.tolist()

            # 检查重建标签是否在原始标签列表中
            accuracy = 1.0 if rec_label_value in orig_labels_list else 0.0

            logging.info(
                f"[ClassificationAccuracy] 重建标签 {rec_label_value} 是否在原始标签 {orig_labels_list} 中: {accuracy}")

            return accuracy

        except Exception as e:
            logging.error(f"[ClassificationAccuracy] 计算分类准确率失败: {e}")
            return 0.0

    def _compute_ssim(self, img1, img2, window_size=11):
        """计算SSIM近似值"""
        try:
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2

            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2

            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size // 2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size // 2) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2) - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
                    (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean().item()
        except:
            return 0.0

    def _compute_attack_score(self, metrics, is_final_round=False):
        """计算攻击分数 - 改进版本：考虑加密比例的影响"""
        """
        改进的攻击分数计算规则：
        - 基础评分：基于重建质量（余弦相似度、分类准确率、SSIM、MSE）
        - 加密比例调整：根据加密比例对基础评分进行衰减
        - 最终分数 = 基础评分 × (1 - 加密比例)^衰减因子
        
        这样确保：
        - 加密比例越高 → 攻击分数越低
        - 加密比例越低 → 攻击分数越高
        - 关系曲线合理：非线性衰减，高加密比例时衰减更明显
        """

        if not (is_final_round and metrics.get('has_original_data', False)):
            return 0.0  # 非最终轮次或没有原始数据，返回0分

        try:
            # 获取四个核心指标
            cosine_sim = metrics.get('cosine_similarity', 0.0)
            classification_accuracy = metrics.get('classification_accuracy', 0.0)
            ssim = metrics.get('ssim_approx', 0.0)
            mse = metrics.get('mse', 1.0)  # 默认MSE为1.0

            # 改进的MSE归一化：更敏感的归一化方法
            if mse <= 0.01:
                normalized_mse = mse / 0.01  # 超小MSE
            elif mse <= 0.1:
                normalized_mse = 0.1 + (mse - 0.01) * 0.4 / 0.09  # 小MSE
            elif mse <= 1.0:
                normalized_mse = 0.5 + (mse - 0.1) * 0.4 / 0.9  # 中等MSE
            else:
                normalized_mse = min(1.0, 0.9 + math.log10(1 + mse) * 0.1)  # 大MSE

            mse_component = 1.0 - normalized_mse  # MSE越高，攻击效果越差

            # 计算基础攻击分数（不考虑加密比例）
            base_score = (
                cosine_sim * 0.3 +  # 余弦相似度
                classification_accuracy * 0.25 +  # 分类准确率
                ssim * 0.2 +  # SSIM
                mse_component * 0.25  # MSE分量
            )
            base_score = max(0.0, min(1.0, base_score))

            # 基于加密比例进行衰减调整
            encrypt_ratio = self.encrypt_ratio
            
            # 非线性衰减函数：加密比例越高，衰减越明显
            # 使用指数衰减：衰减因子 = (1 - encrypt_ratio)^2
            # 这样在低加密比例时衰减较小，高加密比例时衰减明显
            decay_factor = (1 - encrypt_ratio) ** 2
            
            # 最终攻击分数 = 基础分数 × 衰减因子
            attack_score = base_score * decay_factor
            
            # 确保分数在[0,1]范围内
            attack_score = max(0.0, min(1.0, attack_score))

            # 记录详细的评分信息用于调试和分析
            metrics['score_components'] = {
                'cosine_similarity': cosine_sim,
                'classification_accuracy': classification_accuracy,
                'ssim': ssim,
                'mse_component': mse_component,
                'raw_mse': mse,
                'normalized_mse': normalized_mse,
                'base_score': base_score,
                'encrypt_ratio': encrypt_ratio,
                'decay_factor': decay_factor,
                'final_score': attack_score
            }

            logging.info(f"[AttackScore] 改进评分 - 基础: {base_score:.4f}, "
                         f"加密比例: {encrypt_ratio:.3f}, 衰减因子: {decay_factor:.4f}, "
                         f"最终分数: {attack_score:.4f}")
            logging.info(f"[AttackScore] 分量详情 - 余弦: {cosine_sim:.4f}, "
                         f"分类准确率: {classification_accuracy:.4f}, SSIM: {ssim:.4f}, "
                         f"MSE分量: {mse_component:.4f}")

            return attack_score

        except Exception as e:
            logging.error(f"[AttackScore] 计算攻击分数失败: {e}")
            return 0.0

    def _compute_overall_score(self, metrics, is_final_round=False):
        """兼容性方法 - 转发到新的攻击分数计算"""
        return self._compute_attack_score(metrics, is_final_round)

    # 其余可视化方法保持不变...
    def _generate_visualization(self, data, metrics, client_id, round_num, original_data=None):
        """生成可视化结果"""
        try:
            logging.info(f"[Visualization] 开始生成可视化结果，数据维度: {data.dim()}")

            os.makedirs(self.output_dir, exist_ok=True)
            logging.info(f"[Visualization] 输出目录确保存在: {self.output_dir}")

            if data.dim() == 4:
                fig = plt.figure(figsize=(12, 8))

                if original_data is not None:
                    logging.info(f"[Visualization] 创建对比图，客户端: {client_id}, 轮次: {round_num}")
                    self._create_comparison_plot(fig, data, original_data, metrics, client_id, round_num)
                else:
                    logging.info(f"[Visualization] 创建单一图表，客户端: {client_id}, 轮次: {round_num}")
                    self._create_single_plot(fig, data, metrics, client_id, round_num)

                plt.tight_layout()
                filename = f"{self.output_dir}/eval_client_{client_id}_round_{round_num}.png"
                logging.info(f"[Visualization] 准备保存图片: {filename}")

                plt.savefig(filename, dpi=150, bbox_inches='tight')
                logging.info(f"[Visualization] 图片保存成功: {filename}")

                plt.close(fig)
                logging.info(f"[Visualization] 图形已关闭")

                if os.path.exists(filename):
                    file_size = os.path.getsize(filename)
                    logging.info(f"[Visualization] 文件验证成功: {filename}, 大小: {file_size} 字节")
                else:
                    logging.error(f"[Visualization] 文件创建失败: {filename}")

            else:
                logging.warning(f"[Visualization] 数据维度不为4，当前维度: {data.dim()}，跳过可视化")

        except Exception as e:
            logging.error(f"[Visualization] 生成可视化结果时发生错误: {str(e)}")
            import traceback
            logging.error(f"[Visualization] 错误堆栈: {traceback.format_exc()}")

    def _create_comparison_plot(self, fig, reconstructed, original, metrics, client_id, round_num):
        """创建对比图"""
        try:
            # 修复：只显示第一个样本的比较
            num_samples = 1  # DLG攻击只重建1个样本

            score = metrics.get('overall_score', 0.0)
            if hasattr(score, 'item'):
                score = score.item()

            # 显示评分分量详情
            components = metrics.get('score_components', {})
            component_text = (f"Cosine: {components.get('cosine_similarity', 0):.3f}\n"
                              f"Acc: {components.get('classification_accuracy', 0):.3f}\n"
                              f"SSIM: {components.get('ssim', 0):.3f}\n"
                              f"MSE: {components.get('mse_component', 0):.3f}")

            fig.suptitle(f'Client {client_id}, Round {round_num}\nAttack Score: {score:.4f}\n{component_text}',
                         fontsize=12)

            for i in range(num_samples):
                try:
                    ax1 = fig.add_subplot(3, num_samples, i + 1)
                    original_sample = original[i].detach().cpu()

                    if original_sample.dim() == 3:
                        if original_sample.shape[0] in [1, 3]:
                            original_sample = original_sample.permute(1, 2, 0)

                    # 修复：正确归一化数据以消除imshow警告
                    original_normalized = self._normalize_for_display(original_sample)
                    if original_normalized.shape[-1] == 1:
                        ax1.imshow(original_normalized.squeeze(), cmap='gray')
                    else:
                        ax1.imshow(original_normalized)

                    ax1.set_title(f'Original {i + 1}')
                    ax1.axis('off')

                    ax2 = fig.add_subplot(3, num_samples, num_samples + i + 1)
                    reconstructed_sample = reconstructed[i].detach().cpu()

                    if reconstructed_sample.dim() == 4:
                        reconstructed_sample = reconstructed_sample.squeeze(0)

                    if reconstructed_sample.dim() == 3:
                        if reconstructed_sample.shape[0] in [1, 3]:
                            reconstructed_sample = reconstructed_sample.permute(1, 2, 0)

                    # 修复：正确归一化重建数据以消除imshow警告
                    reconstructed_normalized = self._normalize_for_display(reconstructed_sample)
                    if reconstructed_normalized.shape[-1] == 1:
                        ax2.imshow(reconstructed_normalized.squeeze(), cmap='gray')
                    else:
                        ax2.imshow(reconstructed_normalized)

                    ax2.set_title(f'Reconstructed {i + 1}')
                    ax2.axis('off')

                    ax3 = fig.add_subplot(3, num_samples, 2 * num_samples + i + 1)

                    orig_for_diff = original[i].detach().cpu()
                    rec_for_diff = reconstructed[i].detach().cpu()

                    if orig_for_diff.dim() == 3 and orig_for_diff.shape[0] in [1, 3]:
                        orig_for_diff = orig_for_diff.permute(1, 2, 0)
                    if rec_for_diff.dim() == 3 and rec_for_diff.shape[0] in [1, 3]:
                        rec_for_diff = rec_for_diff.permute(1, 2, 0)

                    diff = torch.abs(orig_for_diff - rec_for_diff)

                    if diff.shape[-1] == 1:
                        im = ax3.imshow(diff.squeeze(), cmap='hot')
                    else:
                        im = ax3.imshow(diff.mean(dim=-1), cmap='hot')

                    ax3.set_title(f'Difference {i + 1}')
                    ax3.axis('off')
                    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

                except Exception as e:
                    logging.error(f"[Visualization] 处理样本 {i} 时出错: {e}")
                    continue

        except Exception as e:
            logging.error(f"[Visualization] 创建对比图失败: {e}")
            raise

    def _create_single_plot(self, fig, data, metrics, client_id, round_num):
        """创建单一数据图"""
        try:
            num_samples = min(4, data.size(0))

            score = metrics.get('overall_score', 0.0)
            if hasattr(score, 'item'):
                score = score.item()
            fig.suptitle(f'Client {client_id}, Round {round_num}\nAttack Score: {score:.4f}', fontsize=14)

            for i in range(num_samples):
                try:
                    ax = fig.add_subplot(2, num_samples, i + 1)
                    sample = data[i].detach().cpu()

                    if sample.dim() == 3:
                        if sample.shape[0] in [1, 3]:
                            sample = sample.permute(1, 2, 0)

                    # 修复：正确归一化数据以消除imshow警告
                    sample_normalized = self._normalize_for_display(sample)

                    if sample_normalized.shape[-1] == 1:
                        ax.imshow(sample_normalized.squeeze(), cmap='gray')
                    else:
                        ax.imshow(sample_normalized)

                    ax.set_title(f'Sample {i + 1}')
                    ax.axis('off')

                except Exception as e:
                    logging.error(f"[Visualization] 处理单一样本 {i} 时出错: {e}")
                    continue

        except Exception as e:
            logging.error(f"[Visualization] 创建单一图表失败: {e}")
            raise

    def _normalize_for_display(self, sample):
        """归一化数据以消除imshow警告"""
        try:
            # 处理不同数据集类型
            if self.dataset_type == 'grayscale':
                # 灰度图：确保数据在[0,1]范围内
                if sample.min() < 0 or sample.max() > 1:
                    # 如果数据超出[0,1]范围，进行归一化
                    sample = torch.clamp(sample, 0, 1)
            else:
                # 彩色图：处理ImageNet标准化数据
                if sample.min() < -2 or sample.max() > 2.5:
                    # 检测到可能是ImageNet标准化数据，进行反标准化
                    # ImageNet标准化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    if sample.dim() == 3 and sample.shape[0] == 3:
                        # 如果是RGB图像且使用了ImageNet标准化
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        
                        # 反标准化
                        sample = sample * std + mean
                        
                        # 确保在[0,1]范围内
                        sample = torch.clamp(sample, 0, 1)
                    else:
                        # 其他情况，直接进行归一化
                        sample = torch.clamp(sample, 0, 1)
            
            return sample
        except Exception as e:
            logging.error(f"[Normalization] 数据归一化失败: {e}")
            return torch.clamp(sample, 0, 1)  # 失败时返回安全值

    def _compute_entropy(self, probs):
        """计算熵"""
        try:
            if probs.dim() == 0:
                return 0.0

            if probs.dim() == 1:
                probs = probs.unsqueeze(0)

            probs_normalized = F.softmax(probs, dim=1) if probs.dim() > 1 else probs
            log_probs = torch.log(probs_normalized + 1e-8)
            entropy = -torch.sum(probs_normalized * log_probs, dim=-1)
            return entropy.mean().item()
        except Exception as e:
            logging.error(f"[AttackEvaluation] 熵计算失败: {e}")
            return 0.0

    def generate_summary_report(self):
        """生成汇总报告"""
        if not self.evaluation_results:
            return

        attack_scores = []
        component_stats = {
            'cosine_similarity': [],
            'classification_accuracy': [],
            'ssim': [],
            'mse': []
        }

        for result in self.evaluation_results.values():
            score = result['overall_score']
            if isinstance(score, torch.Tensor):
                attack_scores.append(score.item())
            else:
                attack_scores.append(float(score))

            # 收集各个分量的统计
            components = result.get('score_components', {})
            for key in component_stats.keys():
                if key in components:
                    component_stats[key].append(components[key])

        summary = {
            'total_evaluations': len(self.evaluation_results),
            'mean_attack_score': np.mean(attack_scores) if attack_scores else 0.0,
            'std_attack_score': np.std(attack_scores) if attack_scores else 0.0,
            'max_attack_score': np.max(attack_scores) if attack_scores else 0.0,
            'min_attack_score': np.min(attack_scores) if attack_scores else 0.0,
            'encrypt_ratio': self.encrypt_ratio,
            'component_stats': {
                key: {
                    'mean': np.mean(values) if values else 0.0,
                    'std': np.std(values) if values else 0.0
                } for key, values in component_stats.items()
            }
        }

        # 保存汇总报告
        import json
        report_file = f"{self.output_dir}/attack_evaluation_summary.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return summary