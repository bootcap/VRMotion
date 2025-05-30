# utils/visualization_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import cv2
from matplotlib.animation import FuncAnimation
import matplotlib
import easyocr
matplotlib.use('Agg')  # 使用非交互式后端

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loader import MotionDataset
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
from tqdm import tqdm

# ==================== 核心可视化类 ====================
class MotionVisualizer:
    """动作可视化器，用于可视化视频帧和动作数据"""
    
    def __init__(self, dataset: MotionDataset, output_dir: str = "visualization_output", test_mode: bool = False):
        """
        初始化可视化器
        
        Args:
            dataset: MotionDataset实例
            output_dir: 输出目录
            test_mode: 是否为测试模式，测试模式下只处理10帧
        """
        self.dataset = dataset
        self.output_dir = output_dir
        self.test_mode = test_mode
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化EasyOCR
        self.reader = easyocr.Reader(['en'])
        
        # 创建图形
        self.fig = plt.figure(figsize=(20, 10))
        
        # 创建左右两个子图
        self.video_ax = self.fig.add_subplot(121)
        self.motion_ax = self.fig.add_subplot(122, projection='3d')
        
        # 设置标题
        self.fig.suptitle('Motion Visualization', fontsize=16)
        self.video_ax.set_title('Video Frames')
        self.motion_ax.set_title('Motion Data')
        
        # 初始化视频帧显示
        self.video_im = self.video_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.video_ax.axis('off')
        
        # 设置动作图的范围和标签
        self.motion_ax.set_xlim([-1, 1])
        self.motion_ax.set_ylim([-1, 1])
        self.motion_ax.set_zlim([-1, 1])
        self.motion_ax.set_xlabel('Forward (X)')
        self.motion_ax.set_ylabel('Right (Z)')
        self.motion_ax.set_zlabel('Up (Y)')
        
        # 设置视角
        self.motion_ax.view_init(elev=20, azim=45)
        
        # 获取骨骼连接关系
        self.skeleton_connections = get_skeleton_connections()
        
        # 初始化动作线条
        self.motion_lines = []
        if self.skeleton_connections:
            for _ in self.skeleton_connections:
                line, = self.motion_ax.plot([], [], [], color='blue', linewidth=2)
                self.motion_lines.append(line)
        
        # 初始化关节点
        self.joint_points = self.motion_ax.scatter([], [], [], color='red', s=30)
    
    def _extract_timestamp(self, frame):
        """从视频帧中提取时间戳"""
        # 获取左上角区域
        height, width = frame.shape[:2]
        roi = frame[0:100, 0:820]  # 调整ROI大小以适应时间戳区域
        
        # 保存原始ROI用于调试
        debug_dir = os.path.join(self.output_dir, "debug_roi")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, f"roi_{len(os.listdir(debug_dir)):04d}.png"), roi)
        
        # 使用EasyOCR识别文本
        results = self.reader.readtext(roi)
        
        # 打印OCR结果用于调试
        print("OCR Results:", results)
        
        # 查找符合时间戳格式的文本
        for (bbox, text, prob) in results:
            try:
                # 尝试将文本转换为浮点数
                timestamp = float(text)
                if timestamp > 1000000000:  # 确保是Unix时间戳
                    return timestamp
            except:
                continue
        return None
    
    def _update_motion_plot(self, frame_data, joint_names):
        """更新动作图"""
        if frame_data is None or not self.skeleton_connections:
            return
        
        # 清除之前的线条
        for line in self.motion_lines:
            line.set_data([], [])
            line.set_3d_properties([])
        
        # 收集所有关节位置
        joint_positions = []
        for joint_name in joint_names:
            if 'root' not in joint_name.lower():
                joint_idx = joint_names.index(joint_name)
                joint_positions.append(frame_data[joint_idx])
        
        if joint_positions:
            joint_positions = np.array(joint_positions)
            self.joint_points._offsets3d = (joint_positions[:, 0], joint_positions[:, 2], joint_positions[:, 1])
        
        # 绘制新的骨骼连接
        for i, (parent, child) in enumerate(self.skeleton_connections):
            if parent in joint_names and child in joint_names:
                parent_idx = joint_names.index(parent)
                child_idx = joint_names.index(child)
                
                parent_pos = frame_data[parent_idx]
                child_pos = frame_data[child_idx]
                
                self.motion_lines[i].set_data(
                    [parent_pos[0], child_pos[0]],
                    [parent_pos[2], child_pos[2]]
                )
                self.motion_lines[i].set_3d_properties(
                    [parent_pos[1], child_pos[1]]
                )
    
    def visualize_sequence(self, sequence_idx: int, is_train: bool = True):
        """
        可视化一个序列并保存为图片序列
        
        Args:
            sequence_idx: 序列索引
            is_train: 是否从训练集获取数据
        """
        # 获取序列数据
        if is_train:
            sequence = self.dataset.train_data[sequence_idx]
            prefix = "train"
        else:
            sequence = self.dataset.test_data[sequence_idx]
            prefix = "test"
        
        video_frames, input_motion, predict_motion, timestamps = self.dataset.get_sequence(sequence)
        joint_names = timestamps['joint_names']
        
        # 获取帧率
        fps = self.dataset.get_fps() if hasattr(self.dataset, 'get_fps') else 30
        
        # 创建序列输出目录
        sequence_dir = os.path.join(self.output_dir, f"{prefix}_sequence_{sequence_idx}")
        os.makedirs(sequence_dir, exist_ok=True)
        
        # 在测试模式下只处理10帧
        if self.test_mode:
            video_frames = video_frames[:10]
            input_motion = input_motion[:10]
            predict_motion = predict_motion[:10]
            print("Test mode: Processing only first 10 frames")
        
        # 创建两个独立的图形用于保存
        video_fig = plt.figure(figsize=(10, 10))
        video_ax = video_fig.add_subplot(111)
        video_ax.axis('off')
        video_im = video_ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        
        motion_fig = plt.figure(figsize=(10, 10))
        motion_ax = motion_fig.add_subplot(111, projection='3d')
        motion_ax.set_xlim([-1, 1])
        motion_ax.set_ylim([-1, 1])
        motion_ax.set_zlim([-1, 1])
        motion_ax.set_xlabel('Forward (X)')
        motion_ax.set_ylabel('Right (Z)')
        motion_ax.set_zlabel('Up (Y)')
        motion_ax.view_init(elev=20, azim=45)
        
        # 初始化动作线条
        motion_lines = []
        if self.skeleton_connections:
            for _ in self.skeleton_connections:
                line, = motion_ax.plot([], [], [], color='blue', linewidth=2)
                motion_lines.append(line)
        
        # 初始化关节点
        joint_points = motion_ax.scatter([], [], [], color='red', s=30)
        
        # 保存每一帧
        for frame in range(len(video_frames)):
            # 更新视频帧
            video_im.set_array(video_frames[frame])
            
            # 提取视频帧时间戳
            video_timestamp = self._extract_timestamp(video_frames[frame])
            
            # 更新动作数据
            if frame < len(input_motion):
                frame_data = input_motion[frame]
                fbx_timestamp = frame / fps  # 使用数据集帧率
            else:
                # 预测部分，如果有数据则显示
                pred_idx = frame - len(input_motion)
                if pred_idx < len(predict_motion):
                    frame_data = predict_motion[pred_idx]
                    fbx_timestamp = (len(input_motion) + pred_idx) / fps
                else:
                    frame_data = None
                    fbx_timestamp = None
            
            # 更新动作图
            if frame_data is not None and self.skeleton_connections:
                # 清除之前的线条
                for line in motion_lines:
                    line.set_data([], [])
                    line.set_3d_properties([])
                
                # 收集所有关节位置
                joint_positions = []
                for joint_name in joint_names:
                    if 'root' not in joint_name.lower():
                        joint_idx = joint_names.index(joint_name)
                        joint_positions.append(frame_data[joint_idx])
                
                if joint_positions:
                    joint_positions = np.array(joint_positions)
                    joint_points._offsets3d = (joint_positions[:, 0], joint_positions[:, 2], joint_positions[:, 1])
                
                # 绘制新的骨骼连接
                for i, (parent, child) in enumerate(self.skeleton_connections):
                    if parent in joint_names and child in joint_names:
                        parent_idx = joint_names.index(parent)
                        child_idx = joint_names.index(child)
                        
                        parent_pos = frame_data[parent_idx]
                        child_pos = frame_data[child_idx]
                        
                        motion_lines[i].set_data(
                            [parent_pos[0], child_pos[0]],
                            [parent_pos[2], child_pos[2]]
                        )
                        motion_lines[i].set_3d_properties(
                            [parent_pos[1], child_pos[1]]
                        )
            
            # 打印时间戳信息
            print(f"Frame {frame}:")
            print(f"  Video timestamp: {video_timestamp}")
            print(f"  FBX timestamp: {fbx_timestamp}")
            
            # 分别保存视频帧和动作帧
            video_fig.savefig(os.path.join(sequence_dir, f"video_{frame:04d}.png"))
            motion_fig.savefig(os.path.join(sequence_dir, f"motion_{frame:04d}.png"))
            
            print(f"Saved frame {frame} to {sequence_dir}")
        
        # 清理图形
        plt.close(video_fig)
        plt.close(motion_fig)
        
        # 创建视频
        video_path = os.path.join(sequence_dir, "animation.mp4")
        os.system(f"ffmpeg -y -v quiet -framerate {fps} -i {sequence_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {video_path}")
        print(f"Created video at {video_path}")
    
    def visualize_batch(self, batch_size: int = 1, is_train: bool = True):
        """
        可视化一批数据
        
        Args:
            batch_size: 批次大小
            is_train: 是否从训练集获取数据
        """
        if is_train:
            videos, input_motions, predict_motions, timestamps = self.dataset.get_train_batch(batch_size)
        else:
            videos, input_motions, predict_motions, timestamps = self.dataset.get_test_batch(batch_size)
        
        for i in range(batch_size):
            self.visualize_sequence(i, is_train)

# ==================== 独立可视化函数 ====================
def visualize_motion_sequence(
    input_motion: np.ndarray,
    predicted_motion: np.ndarray,
    target_motion: np.ndarray,
    save_path: Optional[str] = None,
    fps: int = 30
) -> None:
    """
    可视化动作序列的预测结果
    
    Args:
        input_motion: 输入动作序列
        predicted_motion: 预测的动作序列
        target_motion: 目标动作序列
        save_path: 保存路径
        fps: 帧率
    """
    # 创建图形
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取骨骼连接
    connections = get_skeleton_connections()
    
    # 创建动画
    def update(frame):
        ax.clear()
        
        # 绘制输入动作
        joints = input_motion[frame]
        for joint, position in joints.items():
            if 'root' not in joint.lower():
                ax.scatter(position[0], position[2], position[1], color='red', s=30)
        
        # 绘制骨骼连接
        for joint1, joint2 in connections:
            if joint1 in joints and joint2 in joints and 'root' not in joint1.lower() and 'root' not in joint2.lower():
                pos1 = joints[joint1]
                pos2 = joints[joint2]
                ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], color='blue')
        
        # 设置固定的坐标轴范围
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # 设置坐标轴标签
        ax.set_xlabel('Forward (X)')
        ax.set_ylabel('Right (Z)')
        ax.set_zlabel('Up (Y)')
        ax.set_title(f'Frame {frame}')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=len(input_motion), interval=1000/fps)
    
    # 保存动画
    if save_path:
        anim.save(save_path, writer='ffmpeg', fps=fps)
    
    plt.close(fig)
    return anim

def visualize_batch_results(
    video_frames: np.ndarray,
    input_motion: np.ndarray,
    predicted_motion: np.ndarray,
    target_motion: np.ndarray,
    save_dir: str,
    batch_idx: int = 0
) -> None:
    """
    可视化批次预测结果
    
    Args:
        video_frames: 视频帧
        input_motion: 输入动作
        predicted_motion: 预测动作
        target_motion: 目标动作
        save_dir: 保存目录
        batch_idx: 批次索引
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 获取当前批次的数据
    video = video_frames[batch_idx]
    input_seq = input_motion[batch_idx]
    pred_seq = predicted_motion[batch_idx]
    target_seq = target_motion[batch_idx]
    
    # 创建图形
    fig = plt.figure(figsize=(10, 10), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # 获取骨骼连接
    connections = get_skeleton_connections()
    
    # 创建临时目录
    temp_dir = os.path.join(save_dir, 'temp_frames')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 渲染每一帧
    for frame_idx in tqdm(range(len(input_seq))):
        ax.clear()
        
        # 获取当前帧的关节位置
        joints = input_seq[frame_idx]
        
        # 绘制关节点
        for joint, position in joints.items():
            if 'root' not in joint.lower():
                ax.scatter(position[0], position[2], position[1], color='red', s=30)
        
        # 绘制骨骼连接
        for joint1, joint2 in connections:
            if joint1 in joints and joint2 in joints and 'root' not in joint1.lower() and 'root' not in joint2.lower():
                pos1 = joints[joint1]
                pos2 = joints[joint2]
                ax.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], color='blue')
        
        # 设置固定的坐标轴范围
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        
        # 设置坐标轴标签
        ax.set_xlabel('Forward (X)')
        ax.set_ylabel('Right (Z)')
        ax.set_zlabel('Up (Y)')
        ax.set_title(f'Frame {frame_idx}')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 保存帧
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path)
        plt.close(fig)
        fig = plt.figure(figsize=(10, 10), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
    
    plt.close(fig)
    
    # 创建视频
    fps = 30
    ffmpeg_cmd = f'ffmpeg -y -v quiet -r {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -preset medium -crf 23 {os.path.join(save_dir, "animation.mp4")}'
    os.system(ffmpeg_cmd)
    

def plot_training_curves(
    train_losses: Dict[str, List[float]],
    val_losses: Optional[Dict[str, List[float]]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    绘制训练曲线
    
    Args:
        train_losses: 训练损失字典
        val_losses: 验证损失字典
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    plt.figure(figsize=(12, 6))
    
    # 绘制训练损失
    for name, values in train_losses.items():
        plt.plot(values, label=f'Train {name}', alpha=0.7)
    
    # 绘制验证损失
    if val_losses:
        for name, values in val_losses.items():
            plt.plot(values, label=f'Val {name}', alpha=0.7)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图像
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show_plot:
        plt.show()
    
    plt.close()

def visualize_prediction_error(
    predicted_motion: np.ndarray,
    target_motion: np.ndarray,
    joint_names: List[str],
    save_path: Optional[str] = None,
    show_plot: bool = True
) -> None:
    """
    可视化预测误差
    
    Args:
        predicted_motion: 预测动作
        target_motion: 目标动作
        joint_names: 关节名称列表
        save_path: 保存路径
        show_plot: 是否显示图像
    """
    # 计算每个关节的误差
    errors = np.linalg.norm(predicted_motion - target_motion, axis=2)  # (seq_len, num_joints)
    mean_errors = np.mean(errors, axis=0)  # (num_joints,)
    
    # 创建柱状图
    plt.figure(figsize=(15, 6))
    bars = plt.bar(range(len(joint_names)), mean_errors)
    
    # 设置标签和标题
    plt.xlabel('Joint')
    plt.ylabel('Mean Error')
    plt.title('Prediction Error by Joint')
    plt.xticks(range(len(joint_names)), joint_names, rotation=45, ha='right')
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # 显示图像
    if show_plot:
        plt.show()
    
    plt.close()

def visualize_prediction_sequence(
    input_motion: np.ndarray,
    predicted_motion: np.ndarray,
    target_motion: np.ndarray,
    save_dir: str,
    fps: int = 30,
    joint_names: List[str] = None
) -> None:
    """
    可视化预测序列，包括输入序列、预测序列和真实序列
    
    Args:
        input_motion: 输入动作序列 (frames, joints, 3)
        predicted_motion: 预测的动作序列 (frames, joints, 3)
        target_motion: 目标动作序列 (frames, joints, 3)
        save_dir: 保存目录
        fps: 帧率
        joint_names: 关节名称列表
    """
    import shutil  # 添加shutil导入
    
    # 如果没有提供关节名称，使用默认的关节名称列表
    if joint_names is None:
        joint_names = [
            'RootNode', 'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End',
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End', 'Spine', 'Spine1',
            'Spine2', 'Neck', 'Neck1', 'Head', 'Head_End', 'RightShoulder', 'RightArm',
            'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
        ]
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5), dpi=100)
    
    # 创建三个子图
    ax1 = fig.add_subplot(131, projection='3d')  # 输入序列
    ax2 = fig.add_subplot(132, projection='3d')  # 预测序列
    ax3 = fig.add_subplot(133, projection='3d')  # 目标序列
    
    # 获取骨骼连接
    connections = get_skeleton_connections()
    
    # 设置标题
    ax1.set_title('Input Motion')
    ax2.set_title('Predicted Motion')
    ax3.set_title('Target Motion')
    
    # 创建临时目录
    temp_dir = os.path.join(save_dir, 'temp_frames')
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 渲染每一帧
    total_frames = len(input_motion) + len(predicted_motion)
    for frame_idx in tqdm(range(total_frames)):
        # 清除所有子图
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # 设置每个子图的属性
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.set_zlim(-1, 1)
            ax.set_xlabel('Forward (X)')
            ax.set_ylabel('Right (Z)')
            ax.set_zlabel('Up (Y)')
            ax.view_init(elev=20, azim=45)
        
        # 绘制输入序列
        if frame_idx < len(input_motion):
            frame_data = input_motion[frame_idx]
            # 绘制关节点
            ax1.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1], color='red', s=30)
            # 绘制骨骼连接
            for joint1, joint2 in connections:
                if joint1 in joint_names and joint2 in joint_names:
                    idx1 = joint_names.index(joint1)
                    idx2 = joint_names.index(joint2)
                    pos1 = frame_data[idx1]
                    pos2 = frame_data[idx2]
                    ax1.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], color='blue')
        
        # 绘制预测序列
        pred_idx = frame_idx - len(input_motion)
        if pred_idx >= 0 and pred_idx < len(predicted_motion):
            frame_data = predicted_motion[pred_idx]
            # 绘制关节点
            ax2.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1], color='green', s=30)
            # 绘制骨骼连接
            for joint1, joint2 in connections:
                if joint1 in joint_names and joint2 in joint_names:
                    idx1 = joint_names.index(joint1)
                    idx2 = joint_names.index(joint2)
                    pos1 = frame_data[idx1]
                    pos2 = frame_data[idx2]
                    ax2.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], color='blue')
        
        # 绘制目标序列
        if pred_idx >= 0 and pred_idx < len(target_motion):
            frame_data = target_motion[pred_idx]
            # 绘制关节点
            ax3.scatter(frame_data[:, 0], frame_data[:, 2], frame_data[:, 1], color='purple', s=30)
            # 绘制骨骼连接
            for joint1, joint2 in connections:
                if joint1 in joint_names and joint2 in joint_names:
                    idx1 = joint_names.index(joint1)
                    idx2 = joint_names.index(joint2)
                    pos1 = frame_data[idx1]
                    pos2 = frame_data[idx2]
                    ax3.plot([pos1[0], pos2[0]], [pos1[2], pos2[2]], [pos1[1], pos2[1]], color='blue')
        
        # 保存帧
        frame_path = os.path.join(temp_dir, f'frame_{frame_idx:04d}.png')
        plt.savefig(frame_path)
    
    plt.close(fig)
    
    # 创建视频
    ffmpeg_cmd = f'ffmpeg -y -v quiet -r {fps} -i {temp_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -preset medium -crf 23 {os.path.join(save_dir, "prediction_animation.mp4")}'
    os.system(ffmpeg_cmd)
    

# ==================== 辅助函数 ====================
def get_skeleton_connections() -> List[Tuple[str, str]]:
    """
    获取骨骼连接关系
    
    Returns:
        骨骼连接列表，每个元素为(父关节, 子关节)的元组
    """
    return [
        ('Hips', 'Spine'),
        ('Spine', 'Spine1'),
        ('Spine1', 'Spine2'),
        ('Spine2', 'Neck'),
        ('Neck', 'Neck1'),
        ('Neck1', 'Head'),
        ('Head', 'Head_End'),
        ('Spine2', 'RightShoulder'),
        ('RightShoulder', 'RightArm'),
        ('RightArm', 'RightForeArm'),
        ('RightForeArm', 'RightHand'),
        ('Spine2', 'LeftShoulder'),
        ('LeftShoulder', 'LeftArm'),
        ('LeftArm', 'LeftForeArm'),
        ('LeftForeArm', 'LeftHand'),
        ('Hips', 'RightUpLeg'),
        ('RightUpLeg', 'RightLeg'),
        ('RightLeg', 'RightFoot'),
        ('RightFoot', 'RightFoot_End'),
        ('Hips', 'LeftUpLeg'),
        ('LeftUpLeg', 'LeftLeg'),
        ('LeftLeg', 'LeftFoot'),
        ('LeftFoot', 'LeftFoot_End')
    ]

# ==================== 测试函数 ====================
def test_visualization():
    """测试可视化功能"""
    # 创建测试数据
    seq_len = 30
    num_joints = 25
    batch_size = 2
    
    # 创建随机动作数据
    input_motion = np.random.randn(seq_len, num_joints, 3)
    predicted_motion = input_motion + np.random.randn(seq_len, num_joints, 3) * 0.1
    target_motion = input_motion + np.random.randn(seq_len, num_joints, 3) * 0.05
    
    # 创建随机视频帧
    video_frames = np.random.randint(0, 255, (batch_size, seq_len, 224, 224, 3), dtype=np.uint8)
    
    # 创建关节名称列表
    joint_names = [
        'RootNode', 'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End',
        'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End', 'Spine', 'Spine1',
        'Spine2', 'Neck', 'Neck1', 'Head', 'Head_End', 'RightShoulder', 'RightArm',
        'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand'
    ]
    
    # 测试动作序列可视化
    print("\nTesting motion sequence visualization...")
    visualize_motion_sequence(
        input_motion,
        predicted_motion,
        target_motion,
        save_path="test_motion.mp4",
        fps=30,
        show_animation=False
    )
    
    # 测试批次结果可视化
    print("\nTesting batch results visualization...")
    visualize_batch_results(
        video_frames,
        np.stack([input_motion] * batch_size),
        np.stack([predicted_motion] * batch_size),
        np.stack([target_motion] * batch_size),
        save_dir="test_batch",
        batch_idx=0,
        fps=30
    )
    
    # 测试训练曲线可视化
    print("\nTesting training curves visualization...")
    train_losses = {
        'total_loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'position_loss': [0.4, 0.3, 0.2, 0.1, 0.05]
    }
    val_losses = {
        'total_loss': [0.6, 0.5, 0.4, 0.3, 0.2],
        'position_loss': [0.5, 0.4, 0.3, 0.2, 0.1]
    }
    plot_training_curves(
        train_losses,
        val_losses,
        save_path="test_curves.png",
        show_plot=False
    )
    
    # 测试预测误差可视化
    print("\nTesting prediction error visualization...")
    visualize_prediction_error(
        predicted_motion,
        target_motion,
        joint_names,
        save_path="test_error.png",
        show_plot=False
    )
    
    print("\nAll visualization tests completed!")

def main():
    """主函数"""
    # 创建数据集
    dataset = MotionDataset(
        data_dir="data",
        input_seconds=8,
        predict_seconds=2,
        fps=8
    )
    
    # 创建可视化器
    visualizer = MotionVisualizer(dataset, output_dir="visualization_output")
    
    # 可视化训练集和测试集的不同序列
    print("\nVisualizing training sequence...")
    if len(dataset.train_data) > 0:
        train_idx = 0  # 使用第一个训练序列
        visualizer.visualize_sequence(train_idx, is_train=True)
        print(f"Visualized training sequence {train_idx}")
    else:
        print("No training sequences available")
    
    print("\nVisualizing test sequence...")
    if len(dataset.test_data) > 0:
        test_idx = 20  # 使用第一个测试序列
        visualizer.visualize_sequence(test_idx, is_train=False)
        print(f"Visualized test sequence {test_idx}")
    else:
        print("No test sequences available")
    
    print("\nVisualization completed!")

if __name__ == '__main__':
    main()