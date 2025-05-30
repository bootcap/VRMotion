import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict

class MotionLoss(nn.Module):
    """综合动作预测损失函数"""
    def __init__(self, 
                 position_weight: float = 1.0,
                 velocity_weight: float = 0.01,
                 acceleration_weight: float = 0.01,
                 displacement_weight: float = 1.0,
                 min_displacement_weight: float = 0.3,
                 skeleton_length_weight: float = 1.0,  # 添加骨架长度保持权重
                 joint_weights: Optional[Dict[str, float]] = None):
        """
        初始化损失函数
        
        Args:
            position_weight: 位置损失权重
            velocity_weight: 速度损失权重
            acceleration_weight: 加速度损失权重
            displacement_weight: 位移损失权重
            min_displacement_weight: 最小位移损失权重
            skeleton_length_weight: 骨架长度保持权重
            joint_weights: 关节权重字典，用于对不同关节的损失进行加权
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight
        self.acceleration_weight = acceleration_weight
        self.displacement_weight = displacement_weight
        self.min_displacement_weight = min_displacement_weight
        self.skeleton_length_weight = skeleton_length_weight  # 添加骨架长度保持权重
        
        # 默认关节权重
        self.joint_weights = joint_weights or {
            'RootNode': 1.0,
            'Hips': 1.0,
            'RightUpLeg': 0.8,
            'RightLeg': 0.8,
            'RightFoot': 1.2,  # 增加脚部权重
            'RightFoot_End': 3,  # 显著增加脚部末端权重
            'LeftUpLeg': 0.8,
            'LeftLeg': 0.8,
            'LeftFoot': 1.2,  # 增加脚部权重
            'LeftFoot_End': 3,  # 显著增加脚部末端权重
            'Spine': 1.0,
            'Spine1': 1.0,
            'Spine2': 1.0,
            'Neck': 1.0,
            'Neck1': 1.0,
            'Head': 1.2,  # 增加头部权重
            'Head_End': 3,  # 显著增加头部末端权重
            'RightShoulder': 0.8,
            'RightArm': 0.8,
            'RightForeArm': 0.8,
            'RightHand': 3,  # 增加手部权重
            'LeftShoulder': 0.8,
            'LeftArm': 0.8,
            'LeftForeArm': 0.8,
            'LeftHand': 3,  # 增加手部权重
        }
        
        # 将关节权重转换为张量
        self.joint_weight_tensor = torch.tensor([self.joint_weights[joint] 
                                               for joint in sorted(self.joint_weights.keys())])
        
        # 定义关节角度限制
        self.joint_limits = {
            'RootNode': {'min': -180, 'max': 180},  # 根节点可以360度旋转
            'Hips': {'min': -45, 'max': 45},  # 髋部前后摆动
            'RightFoot': {'min': -45, 'max': 45},  # 脚踝前后摆动
            'LeftFoot': {'min': -45, 'max': 45},
            'RightFoot_End': {'min': -30, 'max': 30},  # 脚趾弯曲
            'LeftFoot_End': {'min': -30, 'max': 30},
            'Head': {'min': -60, 'max': 60},  # 头部左右转动
            'Head_End': {'min': -45, 'max': 45},  # 头部上下点头
            'RightHand': {'min': -90, 'max': 90},  # 手腕弯曲
            'LeftHand': {'min': -90, 'max': 90},
            'RightArm': {'min': -180, 'max': 180},  # 上臂旋转
            'LeftArm': {'min': -180, 'max': 180},
            'RightForeArm': {'min': -150, 'max': 0},  # 肘部弯曲
            'LeftForeArm': {'min': -150, 'max': 0},
            'RightUpLeg': {'min': -120, 'max': 120},  # 大腿旋转
            'LeftUpLeg': {'min': -120, 'max': 120},
            'RightLeg': {'min': -150, 'max': 0},  # 膝盖弯曲
            'LeftLeg': {'min': -150, 'max': 0},
            'Spine': {'min': -30, 'max': 30},  # 脊椎弯曲
            'Spine1': {'min': -30, 'max': 30},
            'Spine2': {'min': -30, 'max': 30},
            'Neck': {'min': -60, 'max': 60},  # 颈部弯曲
            'Neck1': {'min': -60, 'max': 60},
            'RightShoulder': {'min': -90, 'max': 90},  # 肩部旋转
            'LeftShoulder': {'min': -90, 'max': 90}
        }
        
        # 将角度限制转换为张量
        self.joint_limits_tensor = {}
        for joint, limits in self.joint_limits.items():
            self.joint_limits_tensor[joint] = {
                'min': torch.tensor(limits['min']),
                'max': torch.tensor(limits['max'])
            }
        
        # 定义骨骼连接关系
        self.bone_connections = {
            'Hips': ['RightUpLeg', 'LeftUpLeg', 'Spine'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightFoot_End'],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftFoot_End'],
            'Spine': ['Spine1'],
            'Spine1': ['Spine2'],
            'Spine2': ['Neck'],
            'Neck': ['Neck1'],
            'Neck1': ['Head'],
            'Head': ['Head_End'],
            'Spine2': ['RightShoulder', 'LeftShoulder'],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand']
        }
    
    def compute_joint_angle_loss(self, predicted_motion, target_motion):
        """
        计算关节角度损失
        
        Args:
            predicted_motion: 预测的动作序列 [batch_size, seq_len, num_joints*3]
            target_motion: 目标动作序列 [batch_size, seq_len, num_joints*3]
            
        Returns:
            joint_angle_loss: 关节角度损失
        """
        batch_size, seq_len, _ = predicted_motion.shape
        num_joints = len(self.joint_weights)
        
        # 重塑张量以便处理每个关节
        pred_angles = predicted_motion.view(batch_size, seq_len, num_joints, 3)
        target_angles = target_motion.view(batch_size, seq_len, num_joints, 3)
        
        # 计算每个关节的角度损失
        angle_loss = 0.0
        for i, (joint_name, weight) in enumerate(self.joint_weights.items()):
            # 获取当前关节的预测和目标角度
            pred_joint = pred_angles[:, :, i, :]
            target_joint = target_angles[:, :, i, :]
            
            # 计算角度差异
            angle_diff = torch.abs(pred_joint - target_joint)
            
            # 应用关节权重
            weighted_diff = angle_diff * weight
            
            # 检查是否超出关节限制
            if joint_name in self.joint_limits_tensor:
                limits = self.joint_limits_tensor[joint_name]
                min_limit = limits['min'].to(predicted_motion.device)
                max_limit = limits['max'].to(predicted_motion.device)
                
                # 对超出限制的角度施加惩罚
                below_min = torch.relu(min_limit - pred_joint)
                above_max = torch.relu(pred_joint - max_limit)
                limit_penalty = (below_min + above_max) * 2.0  # 加倍惩罚超出限制的情况
                
                # 累加损失
                angle_loss += (weighted_diff + limit_penalty).mean()
            else:
                # 如果没有定义限制，只使用角度差异
                angle_loss += weighted_diff.mean()
        
        return angle_loss / num_joints  # 归一化损失
    
    def compute_displacement_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算位移损失，确保预测的动作不会过于静态
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            target: 目标动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            displacement_loss: 位移损失
        """
        # 计算相邻帧之间的位移
        pred_displacement = pred[:, 1:] - pred[:, :-1]  # [batch_size, seq_len-1, num_joints, 3]
        target_displacement = target[:, 1:] - target[:, :-1]
        
        # 计算位移差异
        displacement_diff = torch.abs(pred_displacement - target_displacement)  # [batch_size, seq_len-1, num_joints, 3]
        
        # 应用关节权重 - 调整维度以匹配
        joint_weights = self.joint_weight_tensor.to(pred.device)  # [num_joints]
        joint_weights = joint_weights.view(1, 1, -1, 1)  # [1, 1, num_joints, 1]
        weighted_diff = displacement_diff * joint_weights  # [batch_size, seq_len-1, num_joints, 3]
        
        return weighted_diff.mean()
    
    def compute_min_displacement_loss(self, pred: torch.Tensor) -> torch.Tensor:
        """
        计算最小位移损失，惩罚静止不动的情况
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            min_displacement_loss: 最小位移损失
        """
        # 计算每个关节的起始位置和结束位置
        pred_start = pred[:, 0]  # (batch_size, num_joints, 3)
        pred_end = pred[:, -1]
        
        # 计算位移向量的范数
        displacement = pred_end - pred_start
        displacement_norm = torch.norm(displacement, dim=-1)  # (batch_size, num_joints)
        
        # 使用平滑的惩罚函数，当位移接近0时给予较大的惩罚
        min_displacement_loss = torch.exp(-displacement_norm)
        
        # 应用关节权重 - 调整维度以匹配
        joint_weights = self.joint_weight_tensor.to(pred.device)  # [num_joints]
        joint_weights = joint_weights.view(1, -1)  # [1, num_joints]
        weighted_min_displacement = min_displacement_loss * joint_weights
        
        return weighted_min_displacement.mean()
    
    def compute_position_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算位置损失（加权MSE）
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            target: 目标动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            position_loss: 加权位置损失
        """
        # 计算每个关节的MSE
        squared_diff = (pred - target) ** 2  # (batch_size, seq_len, num_joints, 3)
        
        # 应用关节权重 - 调整维度以匹配
        joint_weights = self.joint_weight_tensor.to(pred.device)  # [num_joints]
        joint_weights = joint_weights.view(1, 1, -1, 1)  # [1, 1, num_joints, 1]
        weighted_squared_diff = squared_diff * joint_weights
        
        # 计算平均MSE
        position_loss = weighted_squared_diff.mean()
        return position_loss
    
    def compute_velocity_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算速度损失，确保动作的连续性
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            target: 目标动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            velocity_loss: 速度损失
        """
        # 计算相邻帧之间的位移（速度）
        pred_velocity = pred[:, 1:] - pred[:, :-1]  # (batch_size, seq_len-1, num_joints, 3)
        target_velocity = target[:, 1:] - target[:, :-1]
        
        # 计算速度差异的MSE
        velocity_diff = (pred_velocity - target_velocity) ** 2
        
        # 应用关节权重 - 调整维度以匹配
        joint_weights = self.joint_weight_tensor.to(pred.device)  # [num_joints]
        joint_weights = joint_weights.view(1, 1, -1, 1)  # [1, 1, num_joints, 1]
        weighted_velocity_diff = velocity_diff * joint_weights
        
        # 计算平均速度损失
        velocity_loss = weighted_velocity_diff.mean()
        return velocity_loss
    
    def compute_acceleration_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算加速度损失，使动作更平滑
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            target: 目标动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            acceleration_loss: 加速度损失
        """
        # 计算速度
        pred_velocity = pred[:, 1:] - pred[:, :-1]
        target_velocity = target[:, 1:] - target[:, :-1]
        
        # 计算加速度
        pred_acceleration = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        
        # 计算加速度差异的MSE
        acceleration_diff = (pred_acceleration - target_acceleration) ** 2
        
        # 应用关节权重 - 调整维度以匹配
        joint_weights = self.joint_weight_tensor.to(pred.device)  # [num_joints]
        joint_weights = joint_weights.view(1, 1, -1, 1)  # [1, 1, num_joints, 1]
        weighted_acceleration_diff = acceleration_diff * joint_weights
        
        # 计算平均加速度损失
        acceleration_loss = weighted_acceleration_diff.mean()
        return acceleration_loss
    
    def compute_skeleton_length_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算骨架长度保持损失
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints, 3)
            target: 目标动作序列 (batch_size, seq_len, num_joints, 3)
            
        Returns:
            skeleton_length_loss: 骨架长度保持损失
        """
        batch_size, seq_len, num_joints, _ = pred.shape
        device = pred.device
        
        # 计算目标序列中的骨骼长度
        target_lengths = {}
        for parent, children in self.bone_connections.items():
            parent_idx = list(self.joint_weights.keys()).index(parent)
            for child in children:
                child_idx = list(self.joint_weights.keys()).index(child)
                # 计算目标序列中的骨骼长度
                target_bone = target[:, :, child_idx] - target[:, :, parent_idx]
                target_length = torch.norm(target_bone, dim=-1)  # [batch_size, seq_len]
                target_lengths[(parent, child)] = target_length
        
        # 计算预测序列中的骨骼长度变化
        length_loss = 0.0
        for parent, children in self.bone_connections.items():
            parent_idx = list(self.joint_weights.keys()).index(parent)
            for child in children:
                child_idx = list(self.joint_weights.keys()).index(child)
                # 计算预测序列中的骨骼长度
                pred_bone = pred[:, :, child_idx] - pred[:, :, parent_idx]
                pred_length = torch.norm(pred_bone, dim=-1)  # [batch_size, seq_len]
                
                # 计算长度变化率
                target_length = target_lengths[(parent, child)]
                length_ratio = pred_length / (target_length + 1e-6)  # 避免除零
                
                # 计算长度变化损失（使用平滑的惩罚函数）
                length_diff = torch.abs(length_ratio - 1.0)
                bone_loss = torch.exp(length_diff) - 1.0  # 当长度变化越大，惩罚越重
                
                # 应用关节权重
                parent_weight = self.joint_weights[parent]
                child_weight = self.joint_weights[child]
                weight = (parent_weight + child_weight) / 2.0
                
                length_loss += (bone_loss * weight).mean()
        
        return length_loss / len(self.bone_connections)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Args:
            pred: 预测的动作序列 (batch_size, seq_len, num_joints*3)
            target: 目标动作序列 (batch_size, seq_len, num_joints*3)
            
        Returns:
            losses: 包含各项损失的字典
        """
        # 重塑输入张量
        batch_size, seq_len, _ = pred.shape
        num_joints = 25  # 固定关节数量
        
        # 重塑为 (batch_size, seq_len, num_joints, 3)
        pred = pred.view(batch_size, seq_len, num_joints, 3)
        target = target.view(batch_size, seq_len, num_joints, 3)
        
        # 计算各项损失
        position_loss = self.compute_position_loss(pred, target)
        velocity_loss = self.compute_velocity_loss(pred, target)
        acceleration_loss = self.compute_acceleration_loss(pred, target)
        displacement_loss = self.compute_displacement_loss(pred, target)
        min_displacement_loss = self.compute_min_displacement_loss(pred)
        angle_loss = self.compute_joint_angle_loss(pred.view(batch_size, seq_len, num_joints * 3), 
                                                 target.view(batch_size, seq_len, num_joints * 3))
        skeleton_length_loss = self.compute_skeleton_length_loss(pred, target)
        
        # 计算总损失
        total_loss = (
            self.position_weight * position_loss +
            self.velocity_weight * velocity_loss +
            self.acceleration_weight * acceleration_loss +
            self.displacement_weight * displacement_loss +
            self.min_displacement_weight * min_displacement_loss +
            0.5 * angle_loss +
            self.skeleton_length_weight * skeleton_length_loss
        )
        
        return {
            'total_loss': total_loss,
            'position_loss': position_loss,
            'velocity_loss': velocity_loss,
            'acceleration_loss': acceleration_loss,
            'displacement_loss': displacement_loss,
            'min_displacement_loss': min_displacement_loss,
            'angle_loss': angle_loss,
            'skeleton_length_loss': skeleton_length_loss
        }

def test_loss_function():
    """测试损失函数"""
    # 创建损失函数实例
    loss_fn = MotionLoss()
    
    # 创建测试数据
    batch_size = 2
    seq_len = 20  # 预测序列长度
    num_joints = 25
    pred = torch.randn(batch_size, seq_len, num_joints * 3)  # 展平的动作数据
    target = torch.randn(batch_size, seq_len, num_joints * 3)
    
    # 计算损失
    losses = loss_fn(pred, target)
    
    # 打印结果
    print("\nLoss Function Test Results:")
    print(f"Input shapes:")
    print(f"Pred: {pred.shape}")
    print(f"Target: {target.shape}")
    print(f"\nLoss values:")
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    print(f"Position Loss: {losses['position_loss'].item():.4f}")
    print(f"Velocity Loss: {losses['velocity_loss'].item():.4f}")
    print(f"Acceleration Loss: {losses['acceleration_loss'].item():.4f}")
    print(f"Displacement Loss: {losses['displacement_loss'].item():.4f}")
    print(f"Min Displacement Loss: {losses['min_displacement_loss'].item():.4f}")
    print(f"Angle Loss: {losses['angle_loss'].item():.4f}")
    print(f"Skeleton Length Loss: {losses['skeleton_length_loss'].item():.4f}")

if __name__ == "__main__":
    test_loss_function() 