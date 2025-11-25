import torch
import numpy as np
import logging
import copy
from plato.servers import fedavg_he
from plato.config import Config
from attack_evaluator import AttackEvaluator
import torch.nn.functional as F


class ModelInversionAttack:
    """模型反演攻击类 - 完整修复版本"""

    def __init__(self, model, dataset_config, device='cuda'):
        self.model = model
        self.dataset_config = dataset_config
        self.device = device

        # 确保模型参数启用梯度计算
        self.model.to(device)
        self.model.eval()  # 设置为评估模式
        for param in self.model.parameters():
            param.requires_grad = True

        # 缓存重新排序的模型参数
        self.reordered_params_cache = None

        # 测试模型输出形状
        self._test_model_output_shape()

    def _test_model_output_shape(self):
        """测试模型输出形状"""
        try:
            with torch.no_grad():
                test_input = torch.randn(1, *self.dataset_config['input_shape']).to(self.device)
                test_output = self.model(test_input)
                logging.info(f"[ModelInversion] 模型测试输出形状: {test_output.shape}")
                return test_output.shape
        except Exception as e:
            logging.error(f"[ModelInversion] 模型测试失败: {e}")
            return None

    def dlg_attack(self, gradients, target_class=None, batch_size=1):
        """DLG攻击实现 - 完整修复版本"""
        if gradients is None or len(gradients) == 0:
            logging.warning("[ModelInversion] 没有有效梯度，使用替代重建")
            return self._alternative_reconstruction()

        # 验证和准备真实梯度
        real_gradients = self._prepare_real_gradients(gradients)
        if len(real_gradients) == 0:
            logging.warning("[ModelInversion] 真实梯度为空，使用替代重建")
            return self._alternative_reconstruction()

        # 初始化参数顺序缓存
        self._initialize_parameter_order(real_gradients)

        # 虚拟数据初始化
        dummy_data, dummy_label = self._initialize_dummy_data(batch_size)

        # 优化器配置
        optimizer = torch.optim.AdamW([dummy_data], lr=0.01, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        # 迭代优化过程
        best_loss = float('inf')
        best_data = None
        best_label = None
        patience = 5  # 连续5次损失没有明显改善就提前终止
        no_improve_count = 0
        min_delta = 0.001  # 最小改善阈值

        for step in range(50): 
            optimizer.zero_grad()

            try:
                # 虚拟前向传播
                outputs = self.model(dummy_data)
                num_classes = self.dataset_config['num_classes']

                # 修复：彻底处理输出维度问题
                outputs = self._fix_output_dimensions(outputs, num_classes, batch_size)
                if outputs is None:
                    logging.warning(f"[ModelInversion] 步骤 {step}: 输出维度修复失败，跳过")
                    continue

                # 修复标签格式
                dummy_label_long = self._fix_label_format(dummy_label, batch_size, num_classes)
                if dummy_label_long is None:
                    logging.warning(f"[ModelInversion] 步骤 {step}: 标签格式修复失败，跳过")
                    continue

                # 验证维度匹配
                if not self._validate_dimensions(outputs, dummy_label_long, step):
                    continue

                # 使用安全的交叉熵损失计算
                loss = self._compute_safe_loss(outputs, dummy_label_long, num_classes, step)
                if loss is None:
                    continue

                # 计算虚拟梯度
                dummy_gradients = self._compute_virtual_gradients(loss)
                if dummy_gradients is None:
                    logging.warning("[ModelInversion] 虚拟梯度计算失败，跳过本轮")
                    continue

                # 梯度匹配损失计算
                grad_loss, valid_pairs = self._compute_gradient_matching_loss(dummy_gradients, real_gradients, step)
                if valid_pairs == 0:
                    logging.warning("[ModelInversion] 没有有效的梯度对，跳过本轮")
                    continue

                # 总损失组合
                total_loss = (grad_loss / valid_pairs + 0.05 * loss)

                # 反向传播
                total_loss.backward()               
                optimizer.step()
                scheduler.step()

                # 应用数据约束
                self._apply_data_constraints(dummy_data)

                # 记录最佳结果
                if total_loss.item() < best_loss:
                    # 检查是否显著改善
                    if best_loss - total_loss.item() > min_delta:
                        no_improve_count = 0  # 重置计数器
                    else:
                        no_improve_count += 1
                    
                    best_loss = total_loss.item()
                    best_data = dummy_data.detach().clone()
                    best_label = dummy_label.detach().clone()
                else:
                    no_improve_count += 1
                
                # 提前终止检查
                if no_improve_count >= patience:
                    break

            except Exception as e:
                logging.error(f"[ModelInversion] 步骤 {step} 失败: {e}")
                import traceback
                logging.error(f"[DEBUG] 步骤 {step} 详细错误: {traceback.format_exc()}")
                continue

        # 确保返回有效数据
        if best_data is None:
            logging.warning("[ModelInversion] 没有找到有效的最佳数据，使用替代重建")
            return self._alternative_reconstruction()

        logging.info(f"[ModelInversion] 攻击完成，最佳损失 = {best_loss:.4f}")
        return best_data, best_label

    def _fix_output_dimensions(self, outputs, num_classes, batch_size):
        """修复输出维度问题"""
        try:
            original_shape = outputs.shape
            logging.debug(f"[ModelInversion] 原始输出形状: {original_shape}")

            # 情况1: 输出是1D张量
            if outputs.dim() == 1:
                if outputs.numel() == num_classes:
                    # 单个样本，num_classes个输出
                    outputs = outputs.unsqueeze(0)  # [num_classes] -> [1, num_classes]
                    logging.debug(f"[ModelInversion] 1D输出扩展为2D: {outputs.shape}")
                else:
                    # 无法确定形状，创建新的输出
                    outputs = torch.randn(batch_size, num_classes, device=outputs.device, dtype=outputs.dtype)
                    logging.warning(f"[ModelInversion] 1D输出形状不明确，创建新输出: {outputs.shape}")

            # 情况2: 输出是2D张量但形状不正确
            elif outputs.dim() == 2:
                if outputs.size(1) != num_classes:
                    if outputs.size(1) > num_classes:
                        outputs = outputs[:, :num_classes]
                        logging.debug(f"[ModelInversion] 裁剪输出维度: {outputs.shape}")
                    else:
                        padding = torch.zeros(outputs.size(0), num_classes - outputs.size(1),
                                              device=outputs.device, dtype=outputs.dtype)
                        outputs = torch.cat([outputs, padding], dim=1)
                        logging.debug(f"[ModelInversion] 填充输出维度: {outputs.shape}")

            # 情况3: 输出是3D或更高维
            elif outputs.dim() >= 3:
                # 展平为2D，保留batch维度
                outputs = outputs.view(outputs.size(0), -1)
                logging.debug(f"[ModelInversion] 高维输出展平: {outputs.shape}")

                # 调整到正确的类别数
                if outputs.size(1) != num_classes:
                    if outputs.size(1) > num_classes:
                        outputs = outputs[:, :num_classes]
                    else:
                        padding = torch.zeros(outputs.size(0), num_classes - outputs.size(1),
                                              device=outputs.device, dtype=outputs.dtype)
                        outputs = torch.cat([outputs, padding], dim=1)
                    logging.debug(f"[ModelInversion] 调整后输出形状: {outputs.shape}")

            # 最终验证
            if outputs.dim() != 2 or outputs.size(1) != num_classes:
                logging.error(f"[ModelInversion] 输出形状最终验证失败: {outputs.shape}")
                return None

            logging.debug(f"[ModelInversion] 修复后输出形状: {outputs.shape}")
            return outputs

        except Exception as e:
            logging.error(f"[ModelInversion] 输出维度修复失败: {e}")
            return None

    def _fix_label_format(self, dummy_label, batch_size, num_classes):
        """修复标签格式"""
        try:
            # 确保标签是长整型
            if not isinstance(dummy_label, torch.Tensor):
                dummy_label = torch.tensor(dummy_label, device=self.device, dtype=torch.long)

            # 确保标签是1D
            if dummy_label.dim() > 1:
                dummy_label = dummy_label.flatten()
                logging.debug(f"[ModelInversion] 标签展平: {dummy_label.shape}")

            # 确保标签长度匹配batch_size
            if dummy_label.numel() != batch_size:
                if dummy_label.numel() > batch_size:
                    dummy_label = dummy_label[:batch_size]
                else:
                    # 补充随机标签
                    supplement = torch.randint(0, num_classes, (batch_size - dummy_label.numel(),),
                                               device=self.device, dtype=torch.long)
                    dummy_label = torch.cat([dummy_label, supplement])
                logging.debug(f"[ModelInversion] 调整标签长度: {dummy_label.shape}")

            # 确保标签在有效范围内
            dummy_label = torch.clamp(dummy_label, 0, num_classes - 1)

            logging.debug(f"[ModelInversion] 修复后标签形状: {dummy_label.shape}, 值: {dummy_label.tolist()}")
            return dummy_label

        except Exception as e:
            logging.error(f"[ModelInversion] 标签格式修复失败: {e}")
            return None

    def _validate_dimensions(self, outputs, labels, step):
        """验证维度匹配"""
        try:
            if outputs.dim() != 2:
                logging.error(f"[ModelInversion] 步骤 {step}: 输出应该是2D，实际是{outputs.dim()}D")
                return False

            if labels.dim() != 1:
                logging.error(f"[ModelInversion] 步骤 {step}: 标签应该是1D，实际是{labels.dim()}D")
                return False

            if outputs.size(0) != labels.size(0):
                logging.error(
                    f"[ModelInversion] 步骤 {step}: 批次大小不匹配 输出{outputs.size(0)} vs 标签{labels.size(0)}")
                return False

            logging.debug(f"[ModelInversion] 步骤 {step}: 维度验证通过 - 输出{outputs.shape}, 标签{labels.shape}")
            return True

        except Exception as e:
            logging.error(f"[ModelInversion] 步骤 {step}: 维度验证失败: {e}")
            return False

    def _compute_safe_loss(self, outputs, labels, num_classes, step):
        """安全计算损失 - 修复数据类型问题"""
        try:
            # 确保标签是长整型（交叉熵损失需要long类型）
            if not labels.dtype == torch.long:
                labels = labels.long()
            
            # 确保输出是浮点型
            if not outputs.is_floating_point():
                outputs = outputs.float()
            
            # 方法1: 尝试交叉熵损失
            try:
                loss = F.cross_entropy(outputs, labels)
                logging.debug(f"[ModelInversion] 步骤 {step}: 交叉熵损失 = {loss.item():.4f}")
                return loss
            except Exception as ce_error:
                logging.warning(f"[ModelInversion] 步骤 {step}: 交叉熵损失失败: {ce_error}")

            # 方法2: 使用MSE损失
            try:
                outputs_probs = F.softmax(outputs, dim=1)
                target_probs = F.one_hot(labels, num_classes=num_classes).float()

                # 确保维度匹配
                if outputs_probs.shape != target_probs.shape:
                    logging.warning(
                        f"[ModelInversion] 步骤 {step}: 概率形状不匹配 {outputs_probs.shape} vs {target_probs.shape}")
                    # 调整目标概率形状
                    if target_probs.dim() == 1:
                        target_probs = F.one_hot(labels, num_classes=outputs_probs.size(1)).float()

                loss = F.mse_loss(outputs_probs, target_probs)
                logging.debug(f"[ModelInversion] 步骤 {step}: MSE损失 = {loss.item():.4f}")
                return loss
            except Exception as mse_error:
                logging.error(f"[ModelInversion] 步骤 {step}: MSE损失也失败: {mse_error}")

            # 方法3: 使用KL散度
            try:
                outputs_log_probs = F.log_softmax(outputs, dim=1)
                target_probs = F.one_hot(labels, num_classes=num_classes).float()
                loss = F.kl_div(outputs_log_probs, target_probs, reduction='batchmean')
                logging.debug(f"[ModelInversion] 步骤 {step}: KL损失 = {loss.item():.4f}")
                return loss
            except Exception as kl_error:
                logging.error(f"[ModelInversion] 步骤 {step}: KL损失也失败: {kl_error}")

            return None

        except Exception as e:
            logging.error(f"[ModelInversion] 步骤 {step}: 所有损失计算都失败: {e}")
            return None

    def _compute_gradient_matching_loss(self, dummy_gradients, real_gradients, step):
        """计算梯度匹配损失"""
        grad_loss = 0.0
        valid_pairs = 0

        for i, (d_grad, r_grad) in enumerate(zip(dummy_gradients, real_gradients)):
            if (d_grad is not None and r_grad is not None and
                    d_grad.numel() > 0 and r_grad.numel() > 0):

                # 形状匹配处理
                if d_grad.shape != r_grad.shape:
                    min_elements = min(d_grad.numel(), r_grad.numel())
                    d_flat = d_grad.flatten()[:min_elements]
                    r_flat = r_grad.flatten()[:min_elements]
                    logging.debug(f"[ModelInversion] 步骤 {step}: 梯度 {i} 形状不匹配，使用前{min_elements}个元素")
                else:
                    d_flat = d_grad.flatten()
                    r_flat = r_grad.flatten()

                # 使用L2损失
                pair_loss = F.mse_loss(d_flat, r_flat)
                grad_loss += pair_loss
                valid_pairs += 1
                logging.debug(f"[ModelInversion] 步骤 {step}: 梯度对 {i} 损失 = {pair_loss.item():.6f}")

        logging.debug(
            f"[ModelInversion] 步骤 {step}: 梯度匹配损失总和 = {grad_loss.item():.6f}, 有效对 = {valid_pairs}")
        return grad_loss, valid_pairs

    def _initialize_dummy_data(self, batch_size):
        """虚拟数据初始化 - 增强版本"""
        num_classes = self.dataset_config['num_classes']
        input_shape = self.dataset_config['input_shape']

        # 使用模型的数据类型
        model_dtype = next(self.model.parameters()).dtype

        logging.info(
            f"[ModelInversion] 初始化虚拟数据: batch_size={batch_size}, input_shape={input_shape}, num_classes={num_classes}")

        # 创建虚拟数据
        dummy_data = torch.randn(batch_size, *input_shape, device=self.device, dtype=model_dtype) * 0.1 + 0.5
        dummy_data = torch.clamp(dummy_data, 0, 1)
        dummy_data.requires_grad = True

        # 标签初始化
        dummy_label = torch.randint(0, num_classes, (batch_size,), device=self.device, dtype=torch.long)
        dummy_label = torch.clamp(dummy_label, 0, num_classes - 1)
        dummy_label.requires_grad_(False)

        # 验证
        assert dummy_data.dim() == 4, f"数据应该是4D [B,C,H,W]，实际是{dummy_data.dim()}D"
        assert dummy_label.dim() == 1, f"标签应该是1D，实际是{dummy_label.dim()}D"
        assert dummy_label.size(0) == batch_size, f"标签批次大小不匹配"

        logging.info(f"[ModelInversion] 虚拟数据形状: {dummy_data.shape}, 虚拟标签: {dummy_label.tolist()}")
        return dummy_data, dummy_label

    def _apply_data_constraints(self, dummy_data):
        """数据约束"""
        with torch.no_grad():
            data_range = self.dataset_config.get('data_range', (0, 1))
            dummy_data.data = torch.clamp(dummy_data, data_range[0], data_range[1])

    def _compute_virtual_gradients(self, loss):
        """计算虚拟梯度 - 修复处理占位符参数"""
        try:
            if self.reordered_params_cache is None:
                logging.warning("[ModelInversion] 参数缓存为空")
                return None

            # 区分真实参数和占位符参数
            real_params = []
            placeholder_params = []
            
            for param in self.reordered_params_cache:
                if hasattr(param, '_is_placeholder') and param._is_placeholder:
                    placeholder_params.append(param)
                elif param.requires_grad and param.is_floating_point():
                    real_params.append(param)
                else:
                    # 对于不可训练的参数，创建占位符
                    placeholder = torch.zeros_like(param, dtype=torch.float32, requires_grad=True)
                    placeholder._is_placeholder = True
                    placeholder_params.append(placeholder)

            # 只对真实参数计算梯度
            if not real_params:
                logging.warning("[ModelInversion] 没有真实的可训练参数")
                return None

            dummy_gradients = torch.autograd.grad(
                outputs=loss,
                inputs=real_params,
                grad_outputs=None,
                retain_graph=True,
                create_graph=True,
                allow_unused=True
            )

            # 处理梯度结果
            processed_gradients = []
            real_param_idx = 0
            
            for param in self.reordered_params_cache:
                if hasattr(param, '_is_placeholder') and param._is_placeholder:
                    # 占位符参数，返回零梯度
                    placeholder = torch.zeros_like(param, dtype=torch.float32)
                    processed_gradients.append(placeholder)
                elif param.requires_grad and param.is_floating_point():
                    # 真实参数，获取计算的梯度
                    if real_param_idx < len(dummy_gradients):
                        grad = dummy_gradients[real_param_idx]
                        if grad is not None and grad.is_floating_point():
                            processed_gradients.append(grad)
                        else:
                            placeholder = torch.zeros_like(param, dtype=torch.float32)
                            processed_gradients.append(placeholder)
                    else:
                        placeholder = torch.zeros_like(param, dtype=torch.float32)
                        processed_gradients.append(placeholder)
                    real_param_idx += 1
                else:
                    # 其他情况，返回零梯度
                    placeholder = torch.zeros_like(param, dtype=torch.float32)
                    processed_gradients.append(placeholder)

            logging.debug(f"[ModelInversion] 计算了 {len(processed_gradients)} 个虚拟梯度")
            return processed_gradients

        except Exception as e:
            logging.error(f"[ModelInversion] 虚拟梯度计算错误: {e}")
            return None

    def _initialize_parameter_order(self, real_gradients):
        """初始化参数顺序 - 完全修复动态ResNet分类层兼容性问题"""
        # 获取所有可训练参数，包括动态创建的线性层
        model_params = []
        param_names = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                # 特别处理动态分类层：确保线性层参数被正确识别
                if 'linear' in name and param.shape[0] == self.dataset_config['num_classes']:
                    logging.debug(f"[ModelInversion] 检测到动态分类层参数: {name}, 形状: {param.shape}")
                model_params.append(param)
                param_names.append(name)
        
        # 过滤真实梯度，只保留浮点数类型
        real_gradients = [g for g in real_gradients
                          if g is not None and g.is_floating_point()]

        logging.info(f"[ModelInversion] 模型参数数: {len(model_params)}, 真实梯度数: {len(real_gradients)}")

        # 对于ResNet模型，参数数量不匹配是正常的，因为梯度包含所有层的梯度
        # 而模型参数只包含可训练的参数（不包括num_batches_tracked等）
        
        # 改进的匹配策略：基于参数名称和形状的双重匹配
        reordered_params = []
        used_param_indices = set()
        
        # 首先尝试精确匹配（按顺序）
        for i, grad in enumerate(real_gradients):
            if i < len(model_params):
                # 检查形状是否匹配
                if model_params[i].shape == grad.shape:
                    reordered_params.append(model_params[i])
                    used_param_indices.add(i)
                    logging.debug(f"[ModelInversion] 顺序匹配梯度 {i} 与参数 {i}, 形状: {grad.shape}")
                else:
                    # 形状不匹配，尝试寻找形状匹配的参数
                    found_match = False
                    for j, param in enumerate(model_params):
                        if (j not in used_param_indices and param.shape == grad.shape and
                                param.is_floating_point() and grad.is_floating_point()):
                            reordered_params.append(param)
                            used_param_indices.add(j)
                            found_match = True
                            logging.debug(f"[ModelInversion] 形状匹配梯度 {i} 与参数 {j}, 形状: {grad.shape}")
                            break
                    
                    if not found_match:
                        # 对于无法匹配的梯度，创建占位符参数
                        logging.warning(f"[ModelInversion] 梯度 {i} 形状 {grad.shape} 未找到匹配参数，使用占位符")
                        placeholder = torch.zeros_like(grad, dtype=torch.float32, requires_grad=True)
                        reordered_params.append(placeholder)
            else:
                # 超出模型参数数量，创建占位符
                logging.warning(f"[ModelInversion] 梯度 {i} 超出模型参数数量，使用占位符")
                placeholder = torch.zeros_like(grad, dtype=torch.float32, requires_grad=True)
                reordered_params.append(placeholder)

        # 检查是否有未使用的模型参数
        unused_params = [j for j in range(len(model_params)) if j not in used_param_indices]
        if unused_params:
            logging.warning(f"[ModelInversion] 有 {len(unused_params)} 个模型参数未被使用")
            # 将这些参数添加到末尾
            for j in unused_params:
                reordered_params.append(model_params[j])
                logging.debug(f"[ModelInversion] 添加未使用的参数 {j} 到末尾")

        self.reordered_params_cache = reordered_params
        logging.info(f"[ModelInversion] 参数重新排序完成: {len(reordered_params)} 个参数，匹配了 {len(used_param_indices)} 个模型参数")

    def _prepare_real_gradients(self, gradients):
        """准备真实梯度 - 修复数据类型问题，兼容动态分类层"""
        real_gradients = []
        
        for i, grad in enumerate(gradients):
            if self._is_valid_gradient(grad):
                # 确保梯度在正确的设备上
                if grad.device != self.device:
                    grad = grad.to(self.device)
                
                # 确保梯度是浮点类型
                if not grad.is_floating_point():
                    grad = grad.float()
                    logging.debug(f"[ModelInversion] 梯度 {i}: 转换为浮点类型")
                
                # 特别处理分类层梯度：根据数据集类别数进行验证
                if len(grad.shape) == 2 and grad.shape[0] == self.dataset_config['num_classes']:
                    logging.debug(f"[ModelInversion] 检测到分类层梯度 {i}: 形状 {grad.shape}, 类别数匹配")
                
                real_gradients.append(grad)
                logging.debug(f"[ModelInversion] 梯度 {i}: 形状 {grad.shape}, 类型 {grad.dtype}")
            else:
                logging.warning(f"[ModelInversion] 梯度 {i} 无效，已跳过")

        logging.info(f"[ModelInversion] 准备完成 {len(real_gradients)} 个有效梯度")
        
        # 动态分类层兼容性检查
        if len(real_gradients) > 0:
            # 检查是否有分类层梯度
            classifier_grads = [g for g in real_gradients if len(g.shape) == 2 and g.shape[0] == self.dataset_config['num_classes']]
            if classifier_grads:
                logging.info(f"[ModelInversion] 检测到 {len(classifier_grads)} 个分类层梯度，与数据集类别数 {self.dataset_config['num_classes']} 匹配")
        
        return real_gradients

    def _is_valid_gradient(self, grad):
        """梯度有效性检查"""
        if grad is None or not isinstance(grad, torch.Tensor):
            return False
        if grad.numel() == 0 or not torch.isfinite(grad).all():
            return False
        return True

    def _alternative_reconstruction(self):
        """替代重建方法"""
        input_shape = self.dataset_config['input_shape']
        num_classes = self.dataset_config['num_classes']

        logging.info(f"[ModelInversion] 使用替代重建方法")

        # 创建随机数据
        reconstructed = torch.randn(1, *input_shape, device=self.device) * 0.1
        reconstructed = torch.clamp(reconstructed, 0, 1)

        # 创建随机标签
        labels = torch.randint(0, num_classes, (1,), device=self.device, dtype=torch.long)

        logging.info(f"[ModelInversion] 替代重建数据形状: {reconstructed.shape}, 标签: {labels.item()}")
        return reconstructed, labels


class Server(fedavg_he.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.last_selected_clients = []

        # 模型架构感知
        self.model_type = self._detect_server_model_type()

        # 从配置读取参数
        self.base_encrypt_ratio = getattr(Config().clients, 'encrypt_ratio', 0.1)
        self.label_protection = getattr(Config().clients, 'label_protection', True)
        self.classifier_boost = getattr(Config().clients, 'classifier_encrypt_boost', 1.5)

        # 计算总参数数量
        self.total_parameters = self._calculate_total_parameters()

        # 攻击配置
        self.enable_attack = True
        self.inversion_attacker = None
        self.attack_results = {}

        # 通信开销统计
        self.communication_overhead = {
            'total_bytes_received': 0,
            'client_stats': {},
            'round_stats': {}
        }

        # 本地准确率跟踪
        self.local_accuracies = {}  # 存储每个客户端的本地准确率
        self.final_round_local_accuracy = None  # 最后一轮的平均本地准确率

        # 获取数据集配置
        self.dataset_config = self._get_dataset_config()
        self.attack_evaluator = AttackEvaluator(
            self.dataset_config,
            self.base_encrypt_ratio,
            output_dir="./attack_evaluation"
        )

        logging.info(f"[FedAHPIP] Server 初始化: 模型类型={self.model_type}, "
                     f"总参数={self.total_parameters}, 基础加密比例={self.base_encrypt_ratio}")

    def _detect_server_model_type(self):
        """服务器端模型类型检测"""
        model_name = Config().trainer.model_name.lower()

        if 'lenet' in model_name:
            return 'lenet'
        elif 'resnet' in model_name:
            return 'resnet'
        elif any(name in model_name for name in ['vgg', 'alexnet']):
            return 'cnn'
        else:
            return 'generic'

    def _calculate_total_parameters(self):
        """计算模型总参数数量 - 动态版本，适配自适应分类层"""
        # 动态计算当前模型的实际参数数量
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'model'):
            try:
                model = self.trainer.model
                total_params = sum(p.numel() for p in model.parameters())
                logging.info(f"[FedAHPIP] 服务器端动态计算模型参数: {total_params}")
                return total_params
            except Exception as e:
                logging.warning(f"[FedAHPIP] 服务器端动态计算模型参数失败: {e}")

        # 基于模型类型的回退值 - 使用实际计算值
        if self.model_type == 'lenet':
            # 动态计算LeNet5参数数量
            try:
                from plato.models.registry import get
                temp_model = get(model_name="lenet5", num_classes=getattr(Config().parameters.model, 'num_classes', 6))
                total_params = sum(p.numel() for p in temp_model.parameters())
                logging.info(f"[FedAHPIP] 服务器端动态计算LeNet5参数: {total_params}")
                return total_params
            except Exception as e:
                logging.warning(f"[FedAHPIP] 动态计算LeNet5参数失败: {e}, 使用默认值91606")
                return 91606  # 使用客户端实际计算的值
        elif self.model_type == 'resnet':
            return 11173962
        else:
            return 1000000

    def _get_dataset_config(self):
        """获取数据集配置"""
        datasource = Config().data.datasource.lower()

        configs = {
            'mnist': {
                'name': 'mnist',
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'data_range': (0, 1),
            },
            'cifar10': {
                'name': 'cifar10',
                'input_shape': (3, 32, 32),
                'num_classes': 10,
                'data_range': (0, 1),
            },
            'cifar100': {
                'name': 'cifar100',
                'input_shape': (3, 32, 32),
                'num_classes': 100,
                'data_range': (0, 1),
            },
            'fashion_mnist': {
                'name': 'fashion_mnist',
                'input_shape': (1, 28, 28),
                'num_classes': 10,
                'data_range': (0, 1),
            },
            'neu_cls': {
                'name': 'neu_cls',
                'input_shape': (1, 32, 32),  # NEU_CLS是单通道灰度图像，调整为32x32
                'num_classes': 6,
                'data_range': (0, 1),
            },
            'carparts-50': {
                'name': 'carparts-50',
                'input_shape': (3, 224, 224),  # ResNet-18期望的输入尺寸
                'num_classes': 50,  # Carparts-50数据集有50个类别
                'data_range': (0, 1),
            }
        }

        if 'carparts_50' in datasource.lower():
            logging.info(f"[FedAHPIP] 检测到Carparts-50数据集，使用专用配置: carparts-50")
            return configs.get('carparts-50')
        
        # 然后检查其他配置
        for key, config in configs.items():
            # 改进匹配逻辑：使用部分匹配而非精确匹配
            if key in datasource.lower() or datasource.lower() in key:
                logging.info(f"[FedAHPIP] 使用数据集配置: {key}")
                return config

        # 默认配置
        logging.warning(f"[FedAHPIP] 未找到数据集 {datasource} 的配置，使用默认配置")
        return {
            'name': datasource,
            'input_shape': (3, 32, 32),
            'num_classes': 10,
            'data_range': (0, 1),
        }

    def _initialize_attacker(self):
        """初始化攻击器 - 完整修复版本"""
        if self.inversion_attacker is None:
            try:
                # 确保模型处于正确状态
                model_copy = copy.deepcopy(self.trainer.model)
                model_copy.eval()  # 设置为评估模式

                # 测试模型输出
                with torch.no_grad():
                    test_input = torch.randn(1, *self.dataset_config['input_shape']).to(self.trainer.device)
                    test_output = model_copy(test_input)
                    logging.info(f"[FedAHPIP] 攻击器模型测试输出: {test_output.shape}")

                self.inversion_attacker = ModelInversionAttack(
                    model=model_copy,
                    dataset_config=self.dataset_config,
                    device=self.trainer.device
                )
                logging.info("[FedAHPIP] 模型反演攻击器初始化成功")

            except Exception as e:
                logging.error(f"[FedAHPIP] 攻击器初始化失败: {e}")
                import traceback
                logging.error(f"[FedAHPIP] 详细错误: {traceback.format_exc()}")

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """处理加密数据的权重聚合"""
        self._record_communication_overhead(updates)

        if self.current_round % 2 != 0:
            self._mask_consensus(updates)
            return baseline_weights
        else:
            previous_weights = copy.deepcopy(self.trainer.model.state_dict())
            aggregated_weights = await super().aggregate_weights(
                updates, baseline_weights, weights_received
            )

            # 收集本地准确率
            self._collect_local_accuracies(updates)

            if self.enable_attack and self._is_final_round():
                self._initialize_attacker()
                await self._perform_final_round_inversion_attack(updates, previous_weights)

            return aggregated_weights

    def _collect_local_accuracies(self, updates):
        """收集客户端的本地准确率"""
        current_round_accuracies = []
        
        for update in updates:
            client_id = update.client_id
            
            # 从报告中提取准确率
            if hasattr(update, 'report') and hasattr(update.report, 'accuracy'):
                accuracy = update.report.accuracy
                current_round_accuracies.append(accuracy)
                
                # 更新客户端准确率历史
                if client_id not in self.local_accuracies:
                    self.local_accuracies[client_id] = []
                self.local_accuracies[client_id].append(accuracy)
                
                logging.debug(f"[LocalAccuracy] 客户端 {client_id} 准确率: {accuracy:.4f}")
        
        # 计算当前轮次的平均准确率
        if current_round_accuracies:
            avg_accuracy = sum(current_round_accuracies) / len(current_round_accuracies)
            logging.info(f"[LocalAccuracy] 轮次 {self.current_round} 平均本地准确率: {avg_accuracy:.4f} "
                        f"({len(current_round_accuracies)} 个客户端)")
            
            # 如果是最后一轮，保存最终准确率
            if self._is_final_round():
                self.final_round_local_accuracy = avg_accuracy
                logging.info(f"[LocalAccuracy] 最终轮次平均本地准确率: {avg_accuracy:.4f}")

    def _record_communication_overhead(self, updates):
        """记录通信开销"""
        round_bytes_received = 0

        for update in updates:
            client_id = update.client_id
            actual_received_bytes = self._calculate_payload_size(update.payload)
            round_bytes_received += actual_received_bytes

            if client_id not in self.communication_overhead['client_stats']:
                self.communication_overhead['client_stats'][client_id] = {
                    'total_bytes_sent': 0,
                    'rounds': {}
                }

            self.communication_overhead['client_stats'][client_id]['total_bytes_sent'] += actual_received_bytes
            self.communication_overhead['client_stats'][client_id]['rounds'][self.current_round] = actual_received_bytes

        self.communication_overhead['total_bytes_received'] += round_bytes_received

        if self.current_round not in self.communication_overhead['round_stats']:
            self.communication_overhead['round_stats'][self.current_round] = {
                'bytes_received': 0,
                'client_count': 0
            }

        self.communication_overhead['round_stats'][self.current_round]['bytes_received'] += round_bytes_received
        self.communication_overhead['round_stats'][self.current_round]['client_count'] = len(updates)

        logging.info(f"[CommOverhead] 轮次 {self.current_round} 总接收: {round_bytes_received / 1024 / 1024:.2f} MB")

    def _is_final_round(self):
        """检查是否是最后轮次"""
        try:
            total_rounds = Config().trainer.rounds
            is_final = self.current_round >= total_rounds
            logging.info(
                f"[FedAHPIP] 当前轮次: {self.current_round}, 总轮次: {total_rounds}, 是否最后轮次: {is_final}")
            return is_final
        except Exception as e:
            logging.warning(f"[FedAHPIP] 检查最后轮次失败: {e}")
            return False

    async def _perform_final_round_inversion_attack(self, updates, previous_weights):
        """在最后轮次执行模型反演攻击 - 完整修复版本"""
        logging.info(f"[FinalRoundAttack] 对 {len(updates)} 个客户端进行攻击")

        if self.inversion_attacker is None:
            logging.error("[FinalRoundAttack] 攻击器未初始化，跳过攻击")
            return

        for update in updates:
            client_id = update.client_id

            try:
                logging.info(f"[FinalRoundAttack] 开始处理客户端 {client_id}")

                # 检查客户端是否发送了原始数据
                original_data = None
                original_labels = None

                if hasattr(update.report, 'metadata') and update.report.metadata:
                    metadata = update.report.metadata
                    if metadata.get('is_final_round', False) and 'original_samples' in metadata:
                        original_samples = metadata['original_samples']
                        original_data_np = original_samples.get('data')
                        original_labels_np = original_samples.get('labels')

                        if original_data_np is not None:
                            original_data = torch.from_numpy(original_data_np).to(self.trainer.device)
                            if original_labels_np is not None:
                                original_labels = torch.from_numpy(original_labels_np).to(self.trainer.device)
                            logging.info(f"[FinalRoundAttack] 客户端 {client_id} 有原始数据: {original_data.shape}")

                # 提取梯度
                gradients = self._extract_gradients_from_encrypted_update(update, previous_weights)
                if gradients is None:
                    logging.warning(f"[FinalRoundAttack] 客户端 {client_id} 梯度提取失败")
                    continue

                logging.info(f"[FinalRoundAttack] 客户端 {client_id} 提取到 {len(gradients)} 个梯度")

                # 执行攻击
                reconstructed_data, reconstructed_labels = self.inversion_attacker.dlg_attack(gradients)

                if reconstructed_data is not None and reconstructed_labels is not None:
                    logging.info(
                        f"[FinalRoundAttack] 客户端 {client_id} 攻击完成: 重建数据 {reconstructed_data.shape}, 标签 {reconstructed_labels.shape}")

                    # 评估攻击效果
                    evaluation_result = self.attack_evaluator.evaluate_attack_quality(
                        reconstructed_data=reconstructed_data,
                        reconstructed_labels=reconstructed_labels,
                        original_data=original_data,
                        original_labels=original_labels,
                        client_id=client_id,
                        round_num=self.current_round,
                        is_final_round=True,
                        communication_stats=self.communication_overhead['client_stats'].get(client_id, {})
                    )

                    # 保存结果
                    self.attack_results[client_id] = {
                        'reconstructed_data': reconstructed_data.cpu(),
                        'reconstructed_labels': reconstructed_labels.cpu(),
                        'original_data': original_data.cpu() if original_data is not None else None,
                        'original_labels': original_labels.cpu() if original_labels is not None else None,
                        'evaluation': evaluation_result,
                        'round': self.current_round,
                        'has_original_data': original_data is not None
                    }

                    logging.info(
                        f"[FinalRoundAttack] 客户端 {client_id} 攻击评分: {evaluation_result['overall_score']:.4f}")
                else:
                    logging.warning(f"[FinalRoundAttack] 客户端 {client_id} 攻击返回了空数据")

            except Exception as e:
                logging.error(f"[FinalRoundAttack] 客户端 {client_id} 攻击失败: {e}")
                import traceback
                logging.error(f"[FinalRoundAttack] 详细堆栈: {traceback.format_exc()}")

    def _extract_gradients_from_encrypted_update(self, update, previous_weights):
        """从加密更新中提取梯度 - 确保返回所有梯度"""
        try:
            client_id = getattr(update, 'client_id', 'unknown')
            encrypted_data = update.payload

            logging.info(f"[GradientExtraction] 开始提取客户端 {client_id} 的梯度")

            if not isinstance(encrypted_data, dict):
                logging.warning(f"[GradientExtraction] 客户端 {client_id} payload不是字典类型")
                return None

            # 重构客户端模型
            client_weights = self._reconstruct_client_model(encrypted_data, previous_weights)
            if client_weights is None:
                logging.warning(f"[GradientExtraction] 客户端 {client_id} 模型重构失败")
                return None

            # 计算梯度
            gradients = self._compute_gradients(client_weights, previous_weights, client_id)

            # 过滤None值，但不过滤零梯度
            valid_gradients = [g for g in gradients if g is not None]

            logging.info(f"[FedAHPIP] 客户端 {client_id} - 提取到的梯度: {len(valid_gradients)}/{len(gradients)}")

            if len(valid_gradients) == 0:
                logging.warning(f"[GradientExtraction] 客户端 {client_id} 没有提取到任何梯度")
                return None

            return valid_gradients

        except Exception as e:
            logging.error(f"[GradientExtraction] 梯度提取失败: {e}")
            return None

    def _reconstruct_client_model(self, encrypted_payload, previous_weights):
        """重构客户端模型 - 修复版本"""
        try:
            if not isinstance(encrypted_payload, dict):
                logging.warning("[FedAHPIP] payload不是字典类型")
                return previous_weights

            # 兼容不同字段名：客户端可能发送'unencrypted_weights'或'unencrypted_values'
            unencrypted_values = encrypted_payload.get('unencrypted_values')
            if unencrypted_values is None:
                unencrypted_values = encrypted_payload.get('unencrypted_weights')
            
            # 兼容不同字段名：客户端可能发送'indices'或'unencrypted_indices'
            unencrypted_indices = encrypted_payload.get('unencrypted_indices', [])
            if not unencrypted_indices:
                unencrypted_indices = encrypted_payload.get('indices', [])
            
            # 动态计算当前模型的实际参数数量
            current_total_params = len(self._flatten_weights(previous_weights))
            total_params = encrypted_payload.get('total_params', current_total_params)

            if unencrypted_values is None:
                logging.warning("[FedAHPIP] 未找到未加密的权重值")
                return previous_weights

            # 将previous_weights展平
            previous_flat = self._flatten_weights(previous_weights)

            if len(previous_flat) != total_params:
                logging.warning(f"[FedAHPIP] 参数数量不匹配: 预期{total_params}, 实际{len(previous_flat)}")
                # 使用实际参数数量作为基准
                if len(previous_flat) > total_params:
                    previous_flat = previous_flat[:total_params]
                else:
                    padding = torch.zeros(total_params - len(previous_flat),
                                          device=previous_flat.device, dtype=previous_flat.dtype)
                    previous_flat = torch.cat([previous_flat, padding])

            # 重建客户端权重
            client_flat = previous_flat.clone()

            # 转换未加密值为张量
            if isinstance(unencrypted_values, np.ndarray):
                unencrypted_tensor = torch.from_numpy(unencrypted_values).to(
                    device=previous_flat.device, dtype=previous_flat.dtype
                )
            elif torch.is_tensor(unencrypted_values):
                unencrypted_tensor = unencrypted_values.to(
                    device=previous_flat.device, dtype=previous_flat.dtype
                )
            else:
                logging.warning("[FedAHPIP] 未加密值类型不支持")
                return previous_weights

            # 应用未加密的更新
            if len(unencrypted_indices) > 0 and len(unencrypted_tensor) > 0:
                min_len = min(len(unencrypted_indices), len(unencrypted_tensor))
                if min_len > 0:
                    # 确保indices是numpy数组
                    if isinstance(unencrypted_indices, list):
                        unencrypted_indices = np.array(unencrypted_indices)
                    indices_tensor = torch.from_numpy(unencrypted_indices[:min_len]).to(previous_flat.device)
                    client_flat[indices_tensor] = unencrypted_tensor[:min_len]
                    logging.info(f"[FedAHPIP] 应用了{min_len}个未加密参数更新")
            else:
                logging.warning("[FedAHPIP] 未加密索引或值列表为空")

            # 重新构造成模型结构
            client_weights = self._unflatten_to_model_structure(client_flat, previous_weights)

            logging.info(f"[FedAHPIP] 模型重建完成: {len(unencrypted_indices)}个未加密参数")
            return client_weights

        except Exception as e:
            logging.error(f"[FedAHPIP] 模型重构失败: {e}")
            import traceback
            logging.error(f"[FedAHPIP] 详细错误: {traceback.format_exc()}")
            return previous_weights

    def _compute_gradients(self, client_weights, previous_weights, client_id):
        """计算梯度 - 修复版本：只处理可训练参数"""
        gradients = []

        # 获取所有可训练参数名
        trainable_param_names = []
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                trainable_param_names.append(name)

        logging.info(f"[GradientDebug] 客户端 {client_id} 可训练参数: {len(trainable_param_names)}")

        for key in trainable_param_names:
            if key in client_weights and key in previous_weights:
                client_weight = client_weights[key]
                previous_weight = previous_weights[key]

                if (client_weight.is_floating_point() and
                        previous_weight.is_floating_point() and
                        client_weight.shape == previous_weight.shape):

                    gradient = client_weight - previous_weight

                    # 梯度有效性检查
                    if torch.isfinite(gradient).all() and gradient.numel() > 0:
                        gradients.append(gradient)
                        logging.debug(f"[GradientDebug] 有效梯度层 {key}: {gradient.shape}")
                    else:
                        placeholder = torch.zeros_like(previous_weight)
                        gradients.append(placeholder)
                        logging.warning(f"[GradientDebug] 层 {key} 梯度无效")
                else:
                    placeholder = torch.zeros_like(previous_weights[key])
                    gradients.append(placeholder)
                    logging.warning(f"[GradientDebug] 层 {key} 类型或形状不匹配")
            else:
                placeholder = torch.zeros_like(previous_weights[key])
                gradients.append(placeholder)
                logging.warning(f"[GradientDebug] 层 {key} 在client_weights中不存在")

        logging.info(f"[FedAHPIP] 客户端 {client_id} 有效梯度: {len(gradients)}")
        return gradients

    def _unflatten_to_model_structure(self, flat_vector, original_weight_dict):
        """将扁平向量还原为模型权重结构"""
        reconstructed_weights = {}
        start_idx = 0

        device = next(iter(original_weight_dict.values())).device
        if flat_vector.device != device:
            flat_vector = flat_vector.to(device)

        for key in sorted(original_weight_dict.keys()):
            layer_shape = original_weight_dict[key].shape
            layer_numel = original_weight_dict[key].numel()

            if start_idx + layer_numel > len(flat_vector):
                reconstructed_weights[key] = original_weight_dict[key].clone()
                continue

            layer_weights_flat = flat_vector[start_idx: start_idx + layer_numel]

            if layer_weights_flat.numel() == layer_numel:
                reconstructed_weights[key] = layer_weights_flat.view(layer_shape)
            else:
                reconstructed_weights[key] = original_weight_dict[key].clone()

            start_idx += layer_numel

        return reconstructed_weights

    def _flatten_weights(self, weight_dict):
        """将模型权重字典展平为一维向量"""
        weight_tensors = []
        for key in sorted(weight_dict.keys()):
            weight_tensor = weight_dict[key].view(-1).clone().detach()
            weight_tensors.append(weight_tensor)

        if weight_tensors:
            return torch.cat(weight_tensors)
        else:
            return torch.tensor([], dtype=torch.float32)

    def _mask_consensus(self, updates):
        """掩码共识算法"""
        if not updates:
            self.final_mask = []
            return

        proposals = [update.payload for update in updates if update.payload is not None]
        if not proposals:
            self.final_mask = []
            return

        target_mask_len = int(self.base_encrypt_ratio * self.total_parameters)

        all_indices = []
        for prop in proposals:
            if len(prop) > 0:
                if isinstance(prop, list):
                    tensor_prop = torch.tensor(prop, dtype=torch.long)
                elif isinstance(prop, torch.Tensor):
                    tensor_prop = prop.clone().detach().to(torch.long)
                else:
                    continue
                all_indices.append(tensor_prop)

        if not all_indices:
            self.final_mask = []
            return

        combined_indices = torch.cat(all_indices)
        unique_indices = torch.unique(combined_indices)

        if len(unique_indices) > target_mask_len:
            perm = torch.randperm(len(unique_indices))
            final_mask = unique_indices[perm[:target_mask_len]]
        else:
            final_mask = unique_indices

        if len(final_mask) < target_mask_len:
            all_possible = torch.arange(self.total_parameters, dtype=torch.long)
            mask_set = set(final_mask.tolist())
            unused_indices = torch.tensor([i for i in range(self.total_parameters) if i not in mask_set],
                                          dtype=torch.long)

            if len(unused_indices) > 0:
                supplement_count = min(target_mask_len - len(final_mask), len(unused_indices))
                perm = torch.randperm(len(unused_indices))
                supplement = unused_indices[perm[:supplement_count]]
                final_mask = torch.cat([final_mask, supplement])

        if len(final_mask) > target_mask_len:
            perm = torch.randperm(len(final_mask))
            final_mask = final_mask[perm[:target_mask_len]]

        self.final_mask = final_mask.int().tolist()

        actual_ratio = len(self.final_mask) / self.total_parameters
        logging.info(f"[MaskConsensus] {self.model_type}模型最终掩码: {len(self.final_mask)}/{self.total_parameters}, "
                     f"比例: {actual_ratio:.3f}")

    def choose_clients(self, clients_pool, clients_count):
        """选择客户端"""
        if self.current_round % 2 != 0:
            self.last_selected_clients = super().choose_clients(clients_pool, clients_count)
        return self.last_selected_clients

    def customize_server_payload(self, payload):
        """自定义服务器负载"""
        if self.current_round % 2 != 0:
            return self.encrypted_model
        else:
            return self.final_mask

    def get_logged_items(self):
        """获取日志项 - 新增本地准确率指标"""
        try:
            logged_items = super().get_logged_items()
            logged_items['model_type'] = self.model_type

            # 添加本地准确率指标
            if self.final_round_local_accuracy is not None:
                logged_items['final_local_accuracy'] = f"{self.final_round_local_accuracy:.4f}"
            
            # 添加当前轮次平均本地准确率
            if hasattr(self, 'current_round') and hasattr(self, 'local_accuracies'):
                current_accuracies = []
                for client_id, acc_history in self.local_accuracies.items():
                    if len(acc_history) > 0:
                        current_accuracies.append(acc_history[-1])
                
                if current_accuracies:
                    current_avg_accuracy = sum(current_accuracies) / len(current_accuracies)
                    logged_items['current_local_accuracy'] = f"{current_avg_accuracy:.4f}"

            if hasattr(self, 'communication_overhead'):
                total_bytes = self.communication_overhead.get('total_bytes_received', 0)
                total_mb = total_bytes / (1024 * 1024)
                logged_items['total_comm_mb'] = f"{total_mb:.2f}"

            return logged_items

        except Exception as e:
            logging.error(f"[FedAHPIP] 日志记录错误: {e}")
            return {
                'model_type': self.model_type,
                'round': getattr(self, 'current_round', 0)
            }

    async def wrap_up(self):
        """训练结束时的收尾工作 - 新增本地准确率总结"""
        try:
            # 生成本地准确率总结报告
            self._generate_local_accuracy_summary()
            
            if hasattr(self, 'attack_evaluator') and hasattr(self.attack_evaluator, 'evaluation_results'):
                summary = self.attack_evaluator.generate_summary_report()
                if summary:
                    comm_analysis = self._analyze_communication_overhead()
                    # 修复：使用正确的键名
                    logging.info(f"[AttackEvaluation] 平均攻击评分: {summary['mean_attack_score']:.4f}")

            await super().wrap_up()

        except Exception as e:
            logging.error(f"[SecurityEvaluation] 收尾工作失败: {e}")
            await super().wrap_up()

    def _generate_local_accuracy_summary(self):
        """生成本地准确率总结报告"""
        if not self.local_accuracies:
            logging.warning("[LocalAccuracy] 没有可用的本地准确率数据")
            return

        try:
            # 计算每个客户端的最终准确率
            final_accuracies = {}
            for client_id, acc_history in self.local_accuracies.items():
                if acc_history:
                    final_accuracies[client_id] = acc_history[-1]

            if final_accuracies:
                # 计算统计数据
                acc_values = list(final_accuracies.values())
                avg_accuracy = sum(acc_values) / len(acc_values)
                max_accuracy = max(acc_values)
                min_accuracy = min(acc_values)
                std_accuracy = np.std(acc_values) if len(acc_values) > 1 else 0.0

                # 记录总结信息
                logging.info(f"[LocalAccuracy] 本地准确率总结:")
                logging.info(f"[LocalAccuracy] 客户端数量: {len(final_accuracies)}")
                logging.info(f"[LocalAccuracy] 平均准确率: {avg_accuracy:.4f}")
                logging.info(f"[LocalAccuracy] 最高准确率: {max_accuracy:.4f}")
                logging.info(f"[LocalAccuracy] 最低准确率: {min_accuracy:.4f}")
                logging.info(f"[LocalAccuracy] 标准差: {std_accuracy:.4f}")

                # 保存详细结果到文件
                import json
                import os
                
                summary = {
                    'total_clients': len(final_accuracies),
                    'average_accuracy': avg_accuracy,
                    'max_accuracy': max_accuracy,
                    'min_accuracy': min_accuracy,
                    'std_accuracy': std_accuracy,
                    'client_accuracies': final_accuracies,
                    'all_rounds_data': self.local_accuracies
                }

                # 确保目录存在
                os.makedirs("./results", exist_ok=True)
                
                # 保存到文件
                with open("./results/local_accuracy_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                
                logging.info(f"[LocalAccuracy] 详细结果已保存到: ./results/local_accuracy_summary.json")

        except Exception as e:
            logging.error(f"[LocalAccuracy] 生成总结报告失败: {e}")

    def _analyze_communication_overhead(self):
        """分析通信开销"""
        total_bytes = self.communication_overhead['total_bytes_received']
        total_mb = total_bytes / (1024 * 1024)

        round_count = len(self.communication_overhead.get('round_stats', {}))
        average_per_round = total_mb / round_count if round_count > 0 else 0

        return {
            'total_bytes': total_bytes,
            'total_mb': total_mb,
            'average_per_round_mb': average_per_round,
            'round_count': round_count
        }

    def _calculate_payload_size(self, payload):
        """计算payload大小"""
        try:
            if payload is None:
                return 0

            if isinstance(payload, dict):
                total_size = 0
                for key, value in payload.items():
                    if isinstance(value, np.ndarray):
                        total_size += value.nbytes
                    elif torch.is_tensor(value):
                        total_size += value.nelement() * value.element_size()
                    elif isinstance(value, (list, tuple)):
                        for item in value:
                            total_size += self._calculate_payload_size(item)
                    else:
                        import pickle
                        total_size += len(pickle.dumps(value))
                return total_size

            elif torch.is_tensor(payload):
                return payload.nelement() * payload.element_size()

            else:
                import pickle
                return len(pickle.dumps(payload))

        except Exception as e:
            logging.warning(f"[FedAHPIP] 计算payload大小失败: {e}")
            return 0