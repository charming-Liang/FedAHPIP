"""
A FedAHPIP client with selective homomorphic encryption support for multiple model architectures.
"""
import time
import torch
import torch.nn as nn
import logging
import re
import copy
from plato.clients import simple
from plato.config import Config

class Client(simple.Client):
    """A FedAHPIP client with model-agnostic selective homomorphic encryption support."""

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(
            model=model,
            datasource=datasource,
            algorithm=algorithm,
            trainer=trainer,
            callbacks=callbacks,
        )

        # 从配置文件中读取加密参数
        self.base_encrypt_ratio = Config().clients.encrypt_ratio
        self.label_protection = getattr(Config().clients, 'label_protection', True)
        self.classifier_boost = getattr(Config().clients, 'classifier_encrypt_boost', 1.5)

        # 模型架构检测和层识别（延迟初始化）
        self._model_info_initialized = False  # 标记是否已初始化模型信息
        self.model_type = None
        self.classifier_layers = set()
        self.last_layer_bias = set()

        # 个性化锚定相关参数
        self.hot_parameter_mask = None  # 热参数掩码
        self.layer_protection_ratios = {}  # 分层保护比率
        self.personalized_model_state = None  # 个性化模型状态
        self.stabilization_factor = getattr(Config().clients, 'stabilization_factor', 0.01)  # 稳定因子
        
        # 动量与显著性跟踪
        self.parameter_momentum = {}
        self.parameter_saliency = {}
        self.momentum_alpha = getattr(Config().clients, 'momentum_alpha', 0.9)
        self.saliency_beta = getattr(Config().clients, 'saliency_beta', 0.9)
        self.combination_gamma = getattr(Config().clients, 'combination_gamma', 0.6)

        self.final_mask = None
        self.original_training_samples = None
        self.model_buffer = {}
        self.original_model_state = None
        self.trained_model_state = None
        self.model_updates = None

        # 通信开销跟踪
        self.communication_stats = {
            'total_bytes_sent': 0,
            'total_bytes_received': 0,
            'round_stats': {}
        }
        logging.info(f"[FedAHPIP] Client {self.client_id} 初始化完成，模型信息将延迟加载。")

    def _ensure_model_info_initialized(self):
        """确保模型信息已初始化，如果未初始化则进行检测"""
        if self._model_info_initialized:
            return

        if hasattr(self, 'algorithm') and self.algorithm is not None and hasattr(self.algorithm,
                                                                                 'model') and self.algorithm.model is not None:
            # 执行模型检测和分类层识别
            self.model_type = self._detect_model_type()
            self.classifier_layers = self._identify_classifier_layers()
            self.last_layer_bias = self._identify_last_layer_bias()
            self._model_info_initialized = True
            logging.info(
                f"[FedAHPIP] 客户端 {self.client_id} 模型信息初始化: 模型类型={self.model_type}, 分类层数量={len(self.classifier_layers)}")
            
            # 初始化分层保护比率
            self._initialize_layer_protection_ratios()
        else:
            # 如果algorithm或model不可用，记录警告并使用回退策略
            logging.warning(
                f"[FedAHPIP] 客户端 {self.client_id} 的 algorithm 或 model 不可用，使用回退策略初始化模型信息。")
            self._initialize_with_fallback()

    def _initialize_with_fallback(self):
        """当无法自动检测模型时，使用回退策略初始化模型信息"""
        self.model_type = self._detect_model_type()  # 这通常只依赖配置，可以运行
        # 使用基于模型类型的回退分类层识别
        self.classifier_layers = self._fallback_classifier_identification()
        self.last_layer_bias = self._identify_last_layer_bias_fallback()
        self._model_info_initialized = True
        logging.info(
            f"[FedAHPIP] 客户端 {self.client_id} 使用回退策略初始化模型信息: 模型类型={self.model_type}, 分类层={self.classifier_layers}")
        
        # 初始化分层保护比率
        self._initialize_layer_protection_ratios()

    def _initialize_layer_protection_ratios(self):
        """初始化分层保护比率"""
        if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'model'):
            model = self.algorithm.model
            for name, param in model.named_parameters():
                self.layer_protection_ratios[name] = self._compute_layer_protection_ratio(name, param)
            logging.info(f"[FedAHPIP] 初始化了 {len(self.layer_protection_ratios)} 个层的保护比率")

    def _compute_layer_protection_ratio(self, layer_name, param):
        """计算分层保护比率"""
        base_ratio = self.base_encrypt_ratio
        
        # 确定层类型因子
        layer_type_factor = self._get_layer_type_factor(layer_name)
        
        # 确定关键性因子
        criticality_factor = self._get_criticality_factor(layer_name)
        
        # 计算分层保护比率
        protection_ratio = base_ratio * layer_type_factor * criticality_factor
        
        # 确保比率在合理范围内
        protection_ratio = max(0.0, min(1.0, protection_ratio))
        
        return protection_ratio

    def _get_layer_type_factor(self, layer_name):
        """获取层类型因子"""
        name_lower = layer_name.lower()
        
        # 分类器层：最大保护
        if any(keyword in name_lower for keyword in ['classifier', 'fc', 'linear', 'output', 'head', 'pred', 'logits']):
            return self.classifier_boost  # 通常为1.5
        
        # 最后一层偏置：100%保护
        elif self.label_protection and layer_name in self.last_layer_bias:
            return 1.0 / self.base_encrypt_ratio  # 确保比率为1.0
        
        # 卷积层：适度保护
        elif any(keyword in name_lower for keyword in ['conv', 'features']):
            return 0.8
        
        # 批归一化层：较低保护
        elif any(keyword in name_lower for keyword in ['bn', 'batchnorm', 'normalization']):
            return 0.5
        
        # 默认：基础保护
        else:
            return 1.0

    def _get_criticality_factor(self, layer_name):
        """获取关键性因子"""
        # ResNet模型的分类器层额外增强
        if self.model_type == 'resnet' and layer_name in self.classifier_layers:
            return 1.2
        # 其他架构保持标准
        else:
            return 1.0

    def _identify_last_layer_bias_fallback(self):
        """回退策略：识别最后一层的偏置参数"""
        last_layer_bias = set()
        if self.classifier_layers:
            # 简单地选择分类层中名称排序最后的那个偏置
            classifier_biases = sorted([name for name in self.classifier_layers if 'bias' in name.lower()])
            if classifier_biases:
                last_layer_bias = {classifier_biases[-1]}
        logging.info(f"[FedAHPIP] 回退策略最后一层偏置: {last_layer_bias}")
        return last_layer_bias

    def _detect_model_type(self):
        """自动检测模型类型"""
        model_name = Config().trainer.model_name.lower()

        if 'lenet' in model_name:
            return 'lenet'
        elif 'resnet' in model_name:
            return 'resnet'
        elif any(name in model_name for name in ['vgg', 'alexnet', 'mobilenet', 'efficientnet']):
            return 'cnn'
        elif any(name in model_name for name in ['transformer', 'bert', 'gpt']):
            return 'transformer'
        else:
            return 'generic'

    def _identify_classifier_layers(self):
        """自动识别分类层"""
        classifier_layers = set()

        if hasattr(self, 'algorithm') and hasattr(self.algorithm, 'model'):
            model = self.algorithm.model

            # 通用分类层识别模式
            classifier_patterns = [
                r'classifier', r'fc', r'linear', r'output',
                r'head', r'pred', r'logits', r'proj',
                r'fc\d+', r'linear\d+'  # 匹配fc1, fc2等
            ]

            # 遍历模型的所有参数
            for name, _ in model.named_parameters():
                name_lower = name.lower()

                # 模式匹配
                if any(re.search(pattern, name_lower) for pattern in classifier_patterns):
                    classifier_layers.add(name)

                # 特殊处理：最后一层的权重和偏置
                elif 'weight' in name_lower or 'bias' in name_lower:
                    # 检查是否是最后一层（基于参数形状）
                    param = dict(model.named_parameters())[name]
                    if len(param.shape) == 1 and param.shape[0] == getattr(Config().parameters.model, 'num_classes', 10):
                        classifier_layers.add(name)
                    elif len(param.shape) == 2 and param.shape[0] == getattr(Config().parameters.model, 'num_classes', 10):
                        classifier_layers.add(name)

        # 如果自动识别失败，使用基于模型类型的回退策略
        if not classifier_layers:
            classifier_layers = self._fallback_classifier_identification()

        logging.info(f"[FedAHPIP] 识别到的分类层: {classifier_layers}")
        return classifier_layers

    def _fallback_classifier_identification(self):
        """基于模型类型的回退分类层识别"""
        if self.model_type == 'lenet':
            return {'fc4.weight', 'fc4.bias', 'fc5.weight', 'fc5.bias'}
        elif self.model_type == 'resnet':
            return {'fc.weight', 'fc.bias', 'linear.weight', 'linear.bias'}
        elif self.model_type == 'cnn':
            return {'classifier.weight', 'classifier.bias', 'fc.weight', 'fc.bias'}
        else:
            # 通用模式：选择最后几层作为分类层
            all_params = list(dict(self.algorithm.model.named_parameters()).keys())
            return set(all_params[-4:])  # 取最后4个参数

    def _identify_last_layer_bias(self):
        """识别最后一层的偏置参数"""
        if not self.classifier_layers:
            return set()

        # 查找与类别数相关的偏置参数
        last_layer_bias = set()
        num_classes = getattr(Config().parameters.model, 'num_classes', 10)

        for name, param in self.algorithm.model.named_parameters():
            if 'bias' in name.lower() and name in self.classifier_layers:
                if param.shape == torch.Size([num_classes]):
                    last_layer_bias.add(name)

        # 如果没有找到，选择分类层中的最后一个偏置参数
        if not last_layer_bias and self.classifier_layers:
            classifier_biases = [name for name in self.classifier_layers if 'bias' in name.lower()]
            if classifier_biases:
                last_layer_bias = {classifier_biases[-1]}

        logging.info(f"[FedAHPIP] 最后一层偏置: {last_layer_bias}")
        return last_layer_bias

    def _compute_layer_encryption_ratio(self, layer_name, num_params):
        # 确保模型信息已初始化
        self._ensure_model_info_initialized()
        """计算每层的加密比例"""
        base_ratio = self.base_encrypt_ratio

        # 最后一层偏置：100%加密
        if self.label_protection and layer_name in self.last_layer_bias:
            return 1.0

        # 分类层：增强加密
        elif self.label_protection and layer_name in self.classifier_layers:
            boosted_ratio = min(1.0, base_ratio * self.classifier_boost)

            # 对ResNet等模型的最后一层权重进一步强化
            if self.model_type == 'resnet' and 'weight' in layer_name and layer_name in self.classifier_layers:
                return min(1.0, boosted_ratio * 1.2)
            else:
                return boosted_ratio

        # 卷积层：适度加密
        elif any(keyword in layer_name.lower() for keyword in ['conv', 'features']):
            return base_ratio * 0.8  # 卷积层稍微降低比例

        # 批归一化层：较低加密
        elif any(keyword in layer_name.lower() for keyword in ['bn', 'batchnorm']):
            return base_ratio * 0.5  # BN层加密比例减半

        # 默认：基础比例
        else:
            return base_ratio

    def _classify_layer_type(self, layer_name):
        """对层类型进行分类"""
        name_lower = layer_name.lower()

        if any(keyword in name_lower for keyword in ['conv', 'features']):
            return 'convolutional'
        elif any(keyword in name_lower for keyword in ['fc', 'linear', 'classifier']):
            return 'classifier'
        elif any(keyword in name_lower for keyword in ['bn', 'batchnorm']):
            return 'normalization'
        elif 'bias' in name_lower:
            return 'bias'
        else:
            return 'other'

    def _compute_mask_from_updates(self):
        """基于模型更新的通用分层加密算法"""
        # 确保模型信息已初始化
        self._ensure_model_info_initialized()

        if self.model_updates is None:
            return None

        all_indices = []
        current_idx = 0
        total_params = sum(update.numel() for update in self.model_updates.values())
        target_total_encrypted = int(self.base_encrypt_ratio * total_params)

        layer_stats = {}
        total_encrypted = 0

        for name, update in self.model_updates.items():
            num_params = update.numel()
            flat_update = update.flatten()
            abs_updates = torch.abs(flat_update)

            # 通用分层加密策略
            layer_ratio = self._compute_layer_encryption_ratio(name, num_params)

            # 计算当前层需要加密的参数数量
            mask_len = int(layer_ratio * num_params)
            mask_len = min(mask_len, num_params)

            if mask_len > 0:
                # 选择更新幅度最大的参数进行加密
                _, layer_indices = torch.topk(abs_updates, mask_len)
                global_indices = layer_indices + current_idx
                all_indices.append(global_indices)
                total_encrypted += mask_len

            # 记录层统计信息
            layer_stats[name] = {
                'total_params': num_params,
                'encrypted_params': mask_len,
                'encrypt_ratio': mask_len / num_params if num_params > 0 else 0,
                'layer_type': self._classify_layer_type(name)
            }

            current_idx += num_params

        # 合并结果
        if all_indices:
            mask_indices = torch.unique(torch.cat(all_indices))
            actual_encrypted = len(mask_indices)
        else:
            mask_indices = torch.tensor([], dtype=torch.long)
            actual_encrypted = 0

        actual_ratio = actual_encrypted / total_params if total_params > 0 else 0

        # 比例调整机制
        self._adjust_encryption_ratio(mask_indices, total_params, actual_ratio)

        # 输出分层统计
        self._log_layer_statistics(layer_stats, total_params, actual_encrypted, actual_ratio)

        return mask_indices.tolist()

    def _adjust_encryption_ratio(self, mask_indices, total_params, actual_ratio):
        """调整加密比例以确保接近目标"""
        ratio_tolerance = 0.02

        if abs(actual_ratio - self.base_encrypt_ratio) > ratio_tolerance:
            logging.warning(
                f"[FedAHPIP] 加密比例偏差较大: 实际{actual_ratio:.1%} vs 目标{self.base_encrypt_ratio:.1%}")

            target_count = int(self.base_encrypt_ratio * total_params)
            if len(mask_indices) > target_count:
                perm = torch.randperm(len(mask_indices))
                mask_indices = mask_indices[perm[:target_count]]
                logging.info(f"[FedAHPIP] 调整掩码大小: {len(mask_indices)}/{total_params}")

    def _log_layer_statistics(self, layer_stats, total_params, total_encrypted, actual_ratio):
        """记录分层统计信息"""
        # 按层类型统计
        type_stats = {}
        for stats in layer_stats.values():
            layer_type = stats['layer_type']
            if layer_type not in type_stats:
                type_stats[layer_type] = {'total': 0, 'encrypted': 0}
            type_stats[layer_type]['total'] += stats['total_params']
            type_stats[layer_type]['encrypted'] += stats['encrypted_params']

        for layer_type, stats in type_stats.items():
            if stats['total'] > 0:
                ratio = stats['encrypted'] / stats['total']
                logging.info(f"[LayerType] {layer_type}: {stats['encrypted']}/{stats['total']} ({ratio:.1%})")

        logging.info(f"[FedAHPIP] 总加密: {total_encrypted}/{total_params} ({actual_ratio:.1%})")

    def _update_parameter_importance_metrics(self):
        """更新参数重要性指标（动量和显著性）"""
        if self.trained_model_state is None or self.original_model_state is None:
            return

        # 确保模型信息已初始化
        self._ensure_model_info_initialized()

        for name in self.trained_model_state:
            if name in self.original_model_state:
                # 计算参数更新
                param_update = torch.abs(self.trained_model_state[name] - self.original_model_state[name])
                
                # 更新动量
                if name not in self.parameter_momentum:
                    self.parameter_momentum[name] = param_update
                else:
                    self.parameter_momentum[name] = (self.momentum_alpha * self.parameter_momentum[name] + 
                                                   (1 - self.momentum_alpha) * param_update)

    def _compute_hot_parameter_importance_scores(self):
        """计算热参数重要性分数"""
        importance_scores = {}
        
        for name in self.parameter_momentum:
            if name in self.parameter_saliency:
                # 归一化动量和显著性
                momentum_norm = self.parameter_momentum[name] / (torch.norm(self.parameter_momentum[name]) + 1e-8)
                saliency_norm = self.parameter_saliency[name] / (torch.norm(self.parameter_saliency[name]) + 1e-8)
                
                # 组合分数
                importance_scores[name] = (self.combination_gamma * momentum_norm + 
                                         (1 - self.combination_gamma) * saliency_norm +
                                         self.layer_protection_ratios.get(name, 0.0))
        
        return importance_scores

    def _create_hot_parameter_mask(self, importance_scores):
        """创建热参数掩码"""
        hot_mask = {}
        
        for name, scores in importance_scores.items():
            if name in self.layer_protection_ratios:
                protection_ratio = self.layer_protection_ratios[name]
                num_hot_params = int(protection_ratio * scores.numel())
                
                if num_hot_params > 0:
                    # 选择重要性分数最高的参数作为热参数
                    _, hot_indices = torch.topk(scores.flatten(), num_hot_params)
                    
                    # 创建二进制掩码
                    layer_mask = torch.zeros_like(scores, dtype=torch.bool)
                    layer_mask.view(-1)[hot_indices] = True
                    hot_mask[name] = layer_mask
                else:
                    hot_mask[name] = torch.zeros_like(scores, dtype=torch.bool)
        
        return hot_mask

    def _personalized_anchoring_update(self, global_weights):
        """执行个性化锚定更新"""
        if self.personalized_model_state is None or self.hot_parameter_mask is None:
            logging.warning("[FedAHPIP] 个性化锚定更新条件不满足，使用全局模型")
            return global_weights

        # 确保模型信息已初始化
        self._ensure_model_info_initialized()

        personalized_weights = {}
        
        for name in global_weights:
            if name in self.personalized_model_state and name in self.hot_parameter_mask:
                global_param = global_weights[name]
                local_param = self.personalized_model_state[name]
                hot_mask = self.hot_parameter_mask[name]
                protection_ratio = self.layer_protection_ratios.get(name, 0.0)
                
                # 应用个性化锚定更新规则
                cold_component = global_param * (~hot_mask)
                hot_component = local_param * hot_mask
                stabilization_component = self.stabilization_factor * protection_ratio * hot_mask
                
                personalized_weights[name] = cold_component + hot_component + stabilization_component
                
                logging.debug(f"[PersonalizedAnchoring] 层 {name}: 热参数比例 {hot_mask.float().mean().item():.3f}")
            else:
                # 对于不在热参数掩码中的层，使用全局权重
                personalized_weights[name] = global_weights[name]
        
        logging.info(f"[FedAHPIP] 完成个性化锚定更新")
        return personalized_weights

    def _local_refinement(self, personalized_weights, global_weights, num_epochs=1):
        """执行本地细化"""
        if num_epochs <= 0:
            return personalized_weights

        # 保存当前模型状态
        original_state = self.algorithm.extract_weights()
        
        try:
            # 加载个性化权重
            self.algorithm.load_weights(personalized_weights)
            
            # 配置优化器
            optimizer = torch.optim.SGD(self.algorithm.model.parameters(), lr=0.001)
            
            for epoch in range(num_epochs):
                total_loss = 0.0
                batch_count = 0
                
                for examples, labels in self.train_loader:
                    examples = examples.to(self.device)
                    labels = labels.to(self.device)
                    
                    optimizer.zero_grad()
                    
                    # 前向传播
                    outputs = self.algorithm.model(examples)
                    
                    # 计算任务损失
                    task_loss = torch.nn.functional.cross_entropy(outputs, labels)
                    
                    # 计算KL散度损失（与全局模型的一致性）
                    with torch.no_grad():
                        global_outputs = self._get_global_model_outputs(global_weights, examples)
                    
                    kl_loss = torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(outputs, dim=1),
                        torch.nn.functional.softmax(global_outputs, dim=1),
                        reduction='batchmean'
                    )
                    
                    # 计算正则化损失（保护热参数）
                    reg_loss = 0.0
                    for name, param in self.algorithm.model.named_parameters():
                        if name in self.hot_parameter_mask and name in self.personalized_model_state:
                            protection_ratio = self.layer_protection_ratios.get(name, 0.0)
                            reg_loss += protection_ratio * torch.norm(param - self.personalized_model_state[name])**2
                    
                    # 组合损失
                    total_batch_loss = task_loss + 0.1 * kl_loss + 0.01 * reg_loss
                    total_batch_loss.backward()
                    optimizer.step()
                    
                    total_loss += total_batch_loss.item()
                    batch_count += 1
                
                logging.info(f"[LocalRefinement] 轮次 {epoch+1}/{num_epochs}, 平均损失: {total_loss/batch_count:.4f}")
            
            # 获取细化后的权重
            refined_weights = self.algorithm.extract_weights()
            return refined_weights
            
        finally:
            # 恢复原始状态
            self.algorithm.load_weights(original_state)

    def _get_global_model_outputs(self, global_weights, examples):
        """获取全局模型的输出（用于KL散度计算）"""
        # 保存当前模型状态
        original_state = self.algorithm.extract_weights()
        
        try:
            # 临时加载全局权重
            self.algorithm.load_weights(global_weights)
            
            # 前向传播（不计算梯度）
            with torch.no_grad():
                outputs = self.algorithm.model(examples)
            
            return outputs
        finally:
            # 恢复原始状态
            self.algorithm.load_weights(original_state)

    async def inbound_processed(self, processed_inbound_payload):
        """通讯轮次处理"""
        if self.current_round % 2 != 0:
            return await self._process_odd_round(processed_inbound_payload)
        else:
            return await self._process_even_round(processed_inbound_payload)

    async def _process_odd_round(self, processed_inbound_payload):
        """处理奇数轮次 - 接收全局模型并执行个性化锚定更新"""
        if self.current_round == 1:
            self._save_original_training_samples()

        # 保存原始模型状态
        current_weights = self.algorithm.extract_weights()
        self.original_model_state = {
            name: param.clone().detach() for name, param in current_weights.items()
        }

        # 执行个性化锚定更新
        global_weights = processed_inbound_payload
        
        # 如果是第一轮，初始化个性化模型状态
        if self.personalized_model_state is None:
            self.personalized_model_state = copy.deepcopy(global_weights)
            logging.info(f"[FedAHPIP] 客户端 {self.client_id} 初始化个性化模型状态")

        # 执行个性化锚定更新
        personalized_weights = self._personalized_anchoring_update(global_weights)
        
        # 加载个性化权重进行训练
        self.algorithm.load_weights(personalized_weights)

        # 进行训练
        report, model_weights = await super().inbound_processed(processed_inbound_payload)

        # 保存训练后状态
        trained_weights = self.algorithm.extract_weights()
        self.trained_model_state = {
            name: param.clone().detach() for name, param in trained_weights.items()
        }

        # 更新参数重要性指标
        self._update_parameter_importance_metrics()

        # 计算热参数重要性分数和掩码
        importance_scores = self._compute_hot_parameter_importance_scores()
        self.hot_parameter_mask = self._create_hot_parameter_mask(importance_scores)

        # 更新个性化模型状态为训练后的权重
        self.personalized_model_state = copy.deepcopy(self.trained_model_state)

        self._calculate_model_updates()
        mask_proposal = self._compute_mask_from_updates()

        # 确保掩码提案是列表格式
        if isinstance(mask_proposal, torch.Tensor):
            mask_proposal = mask_proposal.tolist()
        elif mask_proposal is None:
            mask_proposal = []

        # 缓存训练结果
        cached_weights = {}
        for name, param in self.trained_model_state.items():
            cached_weights[name] = param.clone().detach()

        self.model_buffer[self.client_id] = (report, cached_weights)

        # 添加元数据
        self._add_metadata_to_report(report)

        # 计算通信开销
        payload_size = self._calculate_payload_size(mask_proposal)
        self._update_communication_stats(payload_size, 0, self.current_round)

        total_params = sum(update.numel() for update in self.model_updates.values())
        actual_ratio = len(mask_proposal) / total_params if total_params > 0 else 0

        logging.info(f"[FedAHPIP] 客户端 {self.client_id} 生成掩码提案: {len(mask_proposal)}/{total_params} 参数 "
                    f"({actual_ratio:.1%}), 目标比例: {self.base_encrypt_ratio:.1%}")

        return report, mask_proposal

    async def _process_even_round(self, processed_inbound_payload):
        """处理偶数轮次 - 接收最终掩码"""
        # 确保最终掩码是列表格式
        if isinstance(processed_inbound_payload, torch.Tensor):
            self.final_mask = processed_inbound_payload.tolist()
        else:
            self.final_mask = processed_inbound_payload

        # 将最终掩码转换为热参数掩码格式
        self._convert_final_mask_to_hot_parameter_mask()

        if self.client_id in self.model_buffer:
            report, cached_weights = self.model_buffer[self.client_id]
            self._add_metadata_to_report(report)

            # 计算通信开销
            payload_size = self._calculate_payload_size(cached_weights)
            self._update_communication_stats(payload_size, 0, self.current_round)

            return report, cached_weights
        else:
            return await self._fallback_processing(processed_inbound_payload)

    def _convert_final_mask_to_hot_parameter_mask(self):
        """将最终掩码转换为热参数掩码格式"""
        if self.final_mask is None or not hasattr(self, 'algorithm') or not hasattr(self.algorithm, 'model'):
            return

        # 确保模型信息已初始化
        self._ensure_model_info_initialized()

        hot_mask = {}
        current_idx = 0
        
        # 获取模型状态字典
        model_state = self.algorithm.model.state_dict()
        
        for name, param in model_state.items():
            num_params = param.numel()
            
            # 创建该层的掩码
            layer_mask = torch.zeros(num_params, dtype=torch.bool)
            
            # 找出属于该层的掩码索引
            layer_indices = []
            for mask_idx in self.final_mask:
                if current_idx <= mask_idx < current_idx + num_params:
                    layer_indices.append(mask_idx - current_idx)
            
            if layer_indices:
                layer_mask[layer_indices] = True
            
            # 重塑为参数形状
            hot_mask[name] = layer_mask.view(param.shape)
            current_idx += num_params
        
        self.hot_parameter_mask = hot_mask
        logging.info(f"[FedAHPIP] 已将最终掩码转换为热参数掩码，包含 {len(self.hot_parameter_mask)} 个层")

    def _calculate_model_updates(self):
        """计算模型更新"""
        if self.original_model_state is None or self.trained_model_state is None:
            return None

        self.model_updates = {}
        for name in self.original_model_state:
            if name in self.trained_model_state:
                original = self.original_model_state[name]
                trained = self.trained_model_state[name]
                if original.shape == trained.shape:
                    update = trained - original
                    self.model_updates[name] = update

        total_updates = sum(update.numel() for update in self.model_updates.values())
        logging.info(f"[FedAHPIP] Client {self.client_id} 总更新参数: {total_updates}")

        return self.model_updates

    def _save_original_training_samples(self, num_samples=5):
        """保存原始训练数据样本"""
        try:
            # 方法1: 通过train_loader获取数据
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'train_loader'):
                train_loader = self.trainer.train_loader
                if train_loader is not None:
                    for batch_idx, (examples, labels) in enumerate(train_loader):
                        if batch_idx == 0:
                            if examples.size(0) > num_samples:
                                indices = torch.randperm(examples.size(0))[:num_samples]
                                sample_data = examples[indices]
                                sample_labels = labels[indices] if labels is not None else None
                            else:
                                sample_data = examples
                                sample_labels = labels

                            self.original_training_samples = {
                                'data': sample_data.clone().detach(),
                                'labels': sample_labels.clone().detach() if sample_labels is not None else None
                            }
                            return

            # 方法2: 通过datasource获取数据
            if hasattr(self, 'datasource'):
                try:
                    training_dataset = self.datasource.get_train_set()
                    if training_dataset is not None:
                        indices = torch.randperm(len(training_dataset))[:num_samples]
                        sample_data = []
                        sample_labels = []
                        for idx in indices:
                            data, label = training_dataset[idx]
                            sample_data.append(data)
                            sample_labels.append(label)

                        sample_data = torch.stack(sample_data)
                        sample_labels = torch.tensor(sample_labels)

                        self.original_training_samples = {
                            'data': sample_data.clone().detach(),
                            'labels': sample_labels.clone().detach()
                        }
                        return
                except Exception:
                    pass

            # 方法3: 创建模拟数据
            self._create_synthetic_samples(num_samples)

        except Exception:
            self._create_synthetic_samples(num_samples)

    def _create_synthetic_samples(self, num_samples):
        """创建模拟数据"""
        datasource_name = Config().data.datasource.lower()

        if 'mnist' in datasource_name or 'fashion' in datasource_name:
            sample_data = torch.rand(num_samples, 1, 28, 28)
            sample_labels = torch.randint(0, 10, (num_samples,))
        elif 'cifar' in datasource_name:
            sample_data = torch.rand(num_samples, 3, 32, 32)
            sample_labels = torch.randint(0, 10, (num_samples,))
        else:
            sample_data = torch.rand(num_samples, 1, 32, 32)
            sample_labels = torch.randint(0, 10, (num_samples,))

        self.original_training_samples = {
            'data': sample_data,
            'labels': sample_labels
        }

    def _calculate_payload_size(self, payload):
        """计算payload大小（字节）"""
        try:
            if payload is None:
                return 0

            if isinstance(payload, dict):
                # 模型权重字典
                total_size = 0
                for value in payload.values():
                    if hasattr(value, 'nelement') and hasattr(value, 'element_size'):
                        total_size += value.nelement() * value.element_size()
                return total_size
            elif hasattr(payload, 'nelement') and hasattr(payload, 'element_size'):
                # 张量
                return payload.nelement() * payload.element_size()
            elif isinstance(payload, (list, tuple)):
                # 列表或元组，使用pickle估算
                import pickle
                return len(pickle.dumps(payload))
            else:
                import pickle
                return len(pickle.dumps(payload))
        except Exception:
            return 0

    def _update_communication_stats(self, sent_bytes, received_bytes, round_num):
        """更新通信统计"""
        self.communication_stats['total_bytes_sent'] += sent_bytes
        self.communication_stats['total_bytes_received'] += received_bytes

        if round_num not in self.communication_stats['round_stats']:
            self.communication_stats['round_stats'][round_num] = {
                'bytes_sent': 0,
                'bytes_received': 0
            }

        self.communication_stats['round_stats'][round_num]['bytes_sent'] += sent_bytes
        self.communication_stats['round_stats'][round_num]['bytes_received'] += received_bytes

        logging.info(f"[CommOverhead] Client {self.client_id} Round {round_num}: "
                    f"Sent {sent_bytes/1024:.2f} KB")

    def _add_metadata_to_report(self, report):
        """向报告添加元数据 - 确保通信统计正确传输"""
        if not hasattr(report, 'metadata'):
            report.metadata = {}

        is_final_round = self._is_final_round()
        report.metadata['is_final_round'] = is_final_round
        report.metadata['client_id'] = self.client_id

        # 确保通信统计存在且正确
        current_round_stats = self.communication_stats['round_stats'].get(
            self.current_round, {'bytes_sent': 0, 'bytes_received': 0}
        )

        # 添加详细的通信统计
        report.metadata['communication_stats'] = {
            'total_bytes_sent': self.communication_stats['total_bytes_sent'],
            'total_bytes_received': self.communication_stats['total_bytes_received'],
            'current_round_bytes': current_round_stats,
            'all_rounds_stats': self.communication_stats['round_stats']
        }

        # 添加payload大小信息用于验证
        if hasattr(self, 'last_payload_size'):
            report.metadata['payload_size_info'] = {
                'calculated_size': self.last_payload_size,
                'round': self.current_round
            }

        if is_final_round and self.original_training_samples is not None:
            original_data = self.original_training_samples['data'].cpu()
            original_labels = self.original_training_samples['labels'].cpu() if self.original_training_samples[
                                                                                    'labels'] is not None else None

            report.metadata['original_samples'] = {
                'data': original_data.numpy(),
                'labels': original_labels.numpy() if original_labels is not None else None
            }

    def _is_final_round(self):
        """检查是否是最后轮次"""
        try:
            total_rounds = Config().trainer.rounds
            return self.current_round >= total_rounds
        except Exception:
            return False

    async def _fallback_processing(self, processed_inbound_payload):
        """备用处理方案"""
        logging.warning(f"[FedAHPIP] Client {self.client_id} 使用备用处理方案")
        return await super().inbound_processed(processed_inbound_payload)