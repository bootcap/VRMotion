import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import fbx
import os
from tqdm import tqdm
import FbxCommon
from scipy.spatial.transform import Rotation
import pickle
import cv2
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import gc
import hashlib
import json
import csv
import easyocr
from datetime import datetime, timedelta
import re
import time

# 全局调试开关
DEBUG = False
BASE_FBX_FRAME_OFFSET = 10
SKIP_VIDEO_FRAMES_RATIO = 0.05

# 全局最大帧数限制
MAX_FRAMES = int(5e5)

def log_debug(*args, **kwargs):
    """调试日志函数，只有在DEBUG=True时才会输出"""
    if DEBUG:
        print(*args, **kwargs)

def parse_timestamp(timestamp_str: str) -> float:
    """解析时间戳字符串为秒数"""
    try:
        # 尝试解析OCR识别的时间戳
        if '.' in timestamp_str:
            # 将时间戳转换为datetime对象
            dt = datetime.fromtimestamp(float(timestamp_str))
            # 提取时分秒毫秒
            hours = dt.hour
            minutes = dt.minute
            seconds = dt.second
            milliseconds = int(dt.microsecond / 1000)
            # 转换为秒数
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        # 尝试解析CSV中的时间戳格式 (HH:MM:SS:mmm)
        parts = timestamp_str.split(':')
        if len(parts) == 4:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            milliseconds = int(parts[3])
            return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
        
        return None
    except:
        return None

def get_frame_timestamp(video_path: str) -> Tuple[float, int]:
    """获取视频帧的时间戳和对应的帧索引"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 获取原始视频帧率
    
    # 跳过前十分之一的帧
    skip_frames = int(total_frames * SKIP_VIDEO_FRAMES_RATIO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    
    # 初始化OCR
    reader = easyocr.Reader(['en'])
    
    # 尝试读取帧并识别时间戳
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 获取当前帧索引
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 使用OCR识别时间戳
        results = reader.readtext(frame)
        for (bbox, text, prob) in results:
            # 尝试解析时间戳
            timestamp = parse_timestamp(text)
            if timestamp is not None:
                # 将秒数转换为时分秒毫秒格式
                hours = int(timestamp // 3600)
                minutes = int((timestamp % 3600) // 60)
                seconds = int(timestamp % 60)
                milliseconds = int((timestamp % 1) * 1000)
                
                log_debug(f"\n[Video Frame Info]")
                log_debug(f"Frame index: {current_frame}")
                log_debug(f"Original FPS: {original_fps}")
                log_debug(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}")
                log_debug(f"Seconds: {timestamp:.3f}")
                cap.release()
                return timestamp, current_frame, original_fps
    
    cap.release()
    return None, None, None

def get_csv_timestamp(csv_path: str) -> float:
    """从CSV文件获取时间戳"""
    from itertools import islice
    try:
        with open(csv_path, mode="r", encoding="utf-8") as file:
            log_debug(f"\n[CSV File Info]")
            log_debug(f"File: {csv_path}")
            row = next(islice(csv.reader(file), 2, 3), None)
            if row and len(row) >= 2:
                timestamp_str = row[1]
                timestamp = parse_timestamp(timestamp_str)
                if timestamp is not None:
                    hours = int(timestamp // 3600)
                    minutes = int((timestamp % 3600) // 60)
                    seconds = int(timestamp % 60)
                    milliseconds = int((timestamp % 1) * 1000)
                    log_debug(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}:{milliseconds:03d}")
                    log_debug(f"Seconds: {timestamp:.3f}")
                return timestamp
            
    except Exception as e:
        log_debug(f"Error reading CSV file: {e}")
    return None

def calculate_frame_offset(video_timestamp: float, csv_timestamp: float, video_fps: float, fbx_fps: float = 30.0) -> int:
    """计算视频帧和FBX帧之间的偏移"""
    if video_timestamp is None or csv_timestamp is None:
        return 0
    
    # 确保视频和FBX的帧率匹配
    if abs(video_fps - fbx_fps) > 0.1:
        log_debug(f"\n[Frame Rate Warning]")
        log_debug(f"Video FPS ({video_fps}) doesn't match FBX FPS ({fbx_fps})")
        log_debug("This might cause misalignment in frame timing")
    
    # 计算时间差（秒）
    time_diff = video_timestamp - csv_timestamp
    
    # 将时间差转换为帧数（使用FBX的帧率）
    frame_offset = int(time_diff * fbx_fps)
    
    log_debug(f"\n[Frame Offset Calculation]")
    log_debug(f"Video time: {video_timestamp:.3f} seconds")
    log_debug(f"CSV time: {csv_timestamp:.3f} seconds")
    log_debug(f"Time difference: {time_diff:.3f} seconds")
    log_debug(f"Frame offset at {fbx_fps} FPS: {frame_offset} frames")
    
    return frame_offset

def get_data_hash(video_path: str, fbx_path: str, input_frames: int, predict_frames: int, fps: int) -> str:
    """生成数据文件的唯一标识符"""
    # 获取文件的基本信息
    video_stat = os.stat(video_path)
    fbx_stat = os.stat(fbx_path)
    
    # 组合所有相关信息
    info = f"{video_path}:{video_stat.st_mtime}:{fbx_path}:{fbx_stat.st_mtime}:{input_frames}:{predict_frames}:{fps}"
    
    # 生成哈希值
    return hashlib.md5(info.encode()).hexdigest()

class ProcessedData:
    """处理后的数据结构"""
    def __init__(self, video_frames: np.ndarray, input_motion: np.ndarray, 
                 predict_motion: np.ndarray, timestamps: Dict):
        self.video_frames = video_frames
        self.input_motion = input_motion
        self.predict_motion = predict_motion
        self.timestamps = timestamps
    
    def save(self, save_path: str):
        """保存处理后的数据"""
        data = {
            'video_frames': self.video_frames,
            'input_motion': self.input_motion,
            'predict_motion': self.predict_motion,
            'timestamps': self.timestamps
        }
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, load_path: str) -> 'ProcessedData':
        """加载处理后的数据"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return cls(
            video_frames=data['video_frames'],
            input_motion=data['input_motion'],
            predict_motion=data['predict_motion'],
            timestamps=data['timestamps']
        )

class FBXAnimator:
    def __init__(self, fbx_file):
        """初始化FBX动画器，加载文件并计算变换参数"""
        # 检查是否存在对应的pkl文件
        pkl_file = os.path.splitext(fbx_file)[0] + '_fbx.pkl'
        if os.path.exists(pkl_file):
            log_debug(f"Loading existing pkl file: {pkl_file}")
            self._load_from_pkl(pkl_file)
            return
        
        # 如果不存在pkl文件，则处理fbx文件
        self.scene, self.sdk_manager = self._load_fbx_file(fbx_file)
        if not self.scene:
            raise ValueError("Failed to load FBX file")
        
        # 获取动画时间戳信息
        self.animation_timestamps = self._get_animation_timestamps()
        
        # 获取骨骼层次结构
        self.hierarchy, _ = self._get_skeleton_hierarchy()
        
        # 获取第二帧的关节位置用于计算变换参数
        self.second_frame_joints = self._get_joint_positions(1)
        
        # 计算变换参数
        self.transform_params = self._calculate_transform_parameters(self.second_frame_joints)
        
        # 获取动画长度
        self.total_frames = self._get_animation_length()
        
        # 获取所有帧的数据并保存
        self._save_all_frames_data(fbx_file)
    
    def _load_from_pkl(self, pkl_file):
        """从pkl文件加载数据"""
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        self._frames_data = data['frames_data']
        self.hierarchy = data['hierarchy']
        self.transform_params = data['transform_params']
        self.total_frames = data['total_frames']
        self.sdk_manager = None  # 不需要FBX SDK管理器
    
    def _load_fbx_file(self, file_path):
        """加载FBX文件并返回场景"""
        sdk_manager, scene = FbxCommon.InitializeSdkObjects()
        result = FbxCommon.LoadScene(sdk_manager, scene, file_path)
        if not result:
            print("加载FBX文件失败!")
            return None, None
        return scene, sdk_manager
    
    def _get_skeleton_hierarchy(self):
        """从FBX场景提取骨骼层次结构"""
        # 使用固定的骨骼连接关系
        hierarchy = {
            None: ['Hips'],
            'Hips': ['Spine', 'RightUpLeg', 'LeftUpLeg'],
            'Spine': ['Spine1'],
            'Spine1': ['Spine2'],
            'Spine2': ['Neck', 'RightShoulder', 'LeftShoulder'],
            'Neck': ['Neck1'],
            'Neck1': ['Head'],
            'Head': ['Head_End'],
            'RightShoulder': ['RightArm'],
            'RightArm': ['RightForeArm'],
            'RightForeArm': ['RightHand'],
            'LeftShoulder': ['LeftArm'],
            'LeftArm': ['LeftForeArm'],
            'LeftForeArm': ['LeftHand'],
            'RightUpLeg': ['RightLeg'],
            'RightLeg': ['RightFoot'],
            'RightFoot': ['RightFoot_End'],
            'LeftUpLeg': ['LeftLeg'],
            'LeftLeg': ['LeftFoot'],
            'LeftFoot': ['LeftFoot_End']
        }
        
        # 获取所有关节列表
        joint_list = []
        for parent, children in hierarchy.items():
            if parent is not None:
                joint_list.append(parent)
            joint_list.extend(children)
        joint_list = list(set(joint_list))  # 去重
        
        return hierarchy, joint_list
    
    def _get_joint_positions(self, frame, fps=30):
        """获取指定帧的关节位置"""
        root_node = self.scene.GetRootNode()
        joints = {}
        time = fbx.FbxTime()
        
        try:
            time.SetSecondDouble(frame / fps)
        except:
            time.SetFrame(frame)
        
        # 获取时间戳信息
        timestamp = time.GetSecondDouble()
        #log_debug(f"Frame {frame} FBX timestamp: {timestamp}")
        
        def get_node_position(node):
            name = node.GetName()
            is_joint = True
            
            if is_joint:
                try:
                    global_transform = node.EvaluateGlobalTransform(time)
                    translation = global_transform.GetT()
                    joints[name] = np.array([translation[0], translation[1], translation[2]])
                except:
                    joints[name] = np.array([0.0, 0.0, 0.0])
            
            for i in range(node.GetChildCount()):
                get_node_position(node.GetChild(i))
        
        get_node_position(root_node)
        return joints
    
    def _calculate_transform_parameters(self, frame_joints):
        """计算变换参数（旋转、平移、缩放）"""
        key_points = {
            'left_foot': None,
            'right_foot': None,
            'left_shoulder': None,
            'right_shoulder': None,
            'head': None,
            'spine2': None
        }
        
        # 获取关键点
        for name, position in frame_joints.items():
            name_lower = name.lower()
            if 'foot' in name_lower or 'ankle' in name_lower:
                if 'left' in name_lower:
                    key_points['left_foot'] = position
                elif 'right' in name_lower:
                    key_points['right_foot'] = position
            elif 'shoulder' in name_lower or 'clavicle' in name_lower:
                if 'left' in name_lower:
                    key_points['left_shoulder'] = position
                elif 'right' in name_lower:
                    key_points['right_shoulder'] = position
            elif 'head' in name_lower or 'neck' in name_lower:
                key_points['head'] = position
            elif 'spine2' in name_lower:
                key_points['spine2'] = position
        
        # 计算方向向量
        if all(v is not None for v in [key_points['left_foot'], key_points['right_foot'], key_points['head']]):
            forward_vector = key_points['right_foot'] - key_points['left_foot']
            forward_vector = forward_vector / np.linalg.norm(forward_vector)
            
            feet_midpoint = (key_points['left_foot'] + key_points['right_foot']) / 2
            up_vector = key_points['head'] - feet_midpoint
            up_vector = up_vector / np.linalg.norm(up_vector)
            
            right_vector = np.cross(up_vector, forward_vector)
            right_vector = right_vector / np.linalg.norm(right_vector)
        else:
            # 使用默认方向
            forward_vector = np.array([1, 0, 0])
            up_vector = np.array([0, 1, 0])
            right_vector = np.array([0, 0, 1])
        
        # 创建旋转矩阵
        rotation_matrix = np.array([
            forward_vector,
            up_vector,
            right_vector
        ]).T
        
        # 计算缩放因子
        positions = np.array(list(frame_joints.values()))
        non_root_indices = [i for i, name in enumerate(frame_joints.keys()) 
                          if 'root' not in name.lower()]
        
        if non_root_indices:
            # 只使用非root节点的位置
            non_root_positions = positions[non_root_indices]
            
            # 先应用旋转
            rotated_positions = np.dot(non_root_positions, rotation_matrix)
            
            # 计算Y轴范围
            y_min = np.min(rotated_positions[:, 1])
            y_max = np.max(rotated_positions[:, 1])
            y_range = y_max - y_min
            
            # 设置缩放因子，使Y轴范围为[-0.95, 0.95]
            scale_factor = 1.9 / y_range
        else:
            scale_factor = 1.0
            log_debug("Warning: No non-root nodes found. Using default scale factor.")
        
        # 计算平移向量（将spine2平移到原点）
        if key_points['spine2'] is not None:
            # 先应用旋转和缩放到spine2位置
            rotated_spine2 = np.dot(key_points['spine2'], rotation_matrix)
            scaled_spine2 = rotated_spine2 * scale_factor
            # 平移向量应该是spine2位置的负值，这样可以将spine2移到原点
            translation = -scaled_spine2
        else:
            translation = np.zeros(3)
        
        return {
            'rotation_matrix': rotation_matrix,
            'scale_factor': scale_factor,
            'translation': translation
        }
    
    def _get_animation_length(self, default_fps=30):
        """获取动画的帧数"""
        animation_stack_count = self.scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
        if animation_stack_count == 0:
            return min(100, MAX_FRAMES)  # 限制最大帧数
        
        animation_stack = self.scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        time_span = animation_stack.GetLocalTimeSpan()
        start_time = time_span.GetStart()
        end_time = time_span.GetStop()
        
        time_mode = self.scene.GetGlobalSettings().GetTimeMode()
        
        try:
            time = fbx.FbxTime()
            fps = time.GetFrameRate(time_mode)
            duration_seconds = end_time.GetSecondDouble() - start_time.GetSecondDouble()
            frame_count = int(duration_seconds * fps)
        except:
            duration_seconds = end_time.GetSecondDouble() - start_time.GetSecondDouble()
            frame_count = int(duration_seconds * default_fps)
        
        if frame_count <= 0:
            frame_count = 100
        
        # 限制最大帧数为MAX_FRAMES
        frame_count = min(frame_count, MAX_FRAMES)
        if frame_count < MAX_FRAMES:
            log_debug(f"[FBXAnimator] Animation length: {frame_count} frames")
        else:
            log_debug(f"[FBXAnimator] Animation truncated from {frame_count} to {MAX_FRAMES} frames")
        
        return frame_count
    
    def _save_all_frames_data(self, fbx_file):
        """获取所有帧的数据并保存为pkl文件"""
        log_debug("Saving all frames data...")
        self._frames_data = []
        
        # 获取所有帧的数据
        for frame_idx in tqdm(range(self.total_frames)):
            frame_data = self.get_frame_from_fbx(frame_idx)
            self._frames_data.append(frame_data)
        
        # 创建保存路径
        save_path = os.path.splitext(fbx_file)[0] + '_fbx.pkl'
        
        # 保存数据
        save_data = {
            'frames_data': self._frames_data,
            'hierarchy': self.hierarchy,
            'transform_params': self.transform_params,
            'total_frames': self.total_frames
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
        
        log_debug(f"Saved {self.total_frames} frames data to {save_path}")
    
    def get_frame(self, frame_idx):
        """获取指定帧的已处理好的关节点位置"""
        if not hasattr(self, '_frames_data'):
            raise ValueError("No frames data available. Please load a pkl file first.")
        
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.total_frames-1}]")
        
        return self._frames_data[frame_idx]
    
    def get_frame_from_fbx(self, frame_idx, fps=30):
        """从FBX文件获取指定帧的归一化关节位置"""
        # 获取原始关节位置
        joints = self._get_joint_positions(frame_idx, fps)
        
        # 为当前帧计算变换参数
        transform_params = self._calculate_transform_parameters(joints)
        
        # 应用变换
        positions = np.array(list(joints.values()))
        # 1. 先应用旋转
        rotated_positions = np.dot(positions, transform_params['rotation_matrix'])
        # 2. 再应用缩放
        scaled_positions = rotated_positions * transform_params['scale_factor']
        # 3. 最后应用平移
        translated_positions = scaled_positions + transform_params['translation']
        
        # 转换回字典
        normalized_joints = {name: pos for name, pos in zip(joints.keys(), translated_positions)}
        return normalized_joints
    
    def get_skeleton_connections(self):
        """获取骨骼连接关系"""
        connections = []
        for parent, children in self.hierarchy.items():
            if parent is not None:
                for child in children:
                    connections.append((parent, child))
        return connections
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'sdk_manager') and self.sdk_manager is not None:
            self.sdk_manager.Destroy()

    def _get_animation_timestamps(self):
        """获取动画的时间戳信息"""
        if not self.scene:
            return None
            
        # 获取动画堆栈
        anim_stack_count = self.scene.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId))
        if anim_stack_count == 0:
            print("No animation stack found")
            return None
            
        # 获取第一个动画堆栈
        anim_stack = self.scene.GetSrcObject(fbx.FbxCriteria.ObjectType(fbx.FbxAnimStack.ClassId), 0)
        
        # 获取时间信息
        time_span = anim_stack.GetLocalTimeSpan()
        start_time = time_span.GetStart()
        end_time = time_span.GetStop()
        
        # 获取时间模式
        time_mode = self.scene.GetGlobalSettings().GetTimeMode()
        fps = fbx.FbxTime().GetFrameRate(time_mode)
        
        log_debug("\nFBX Animation Time Information:")
        log_debug(f"Start time: {start_time.GetSecondDouble()}")
        log_debug(f"End time: {end_time.GetSecondDouble()}")
        log_debug(f"FPS: {fps}")
        
        # 获取前10帧的时间戳
        log_debug("\nFirst 10 frames timestamps:")
        for frame in range(10):
            time = fbx.FbxTime()
            time.SetSecondDouble(frame / fps)
            log_debug(f"Frame {frame}: {time.GetSecondDouble()}")
        
        return {
            'start_time': start_time.GetSecondDouble(),
            'end_time': end_time.GetSecondDouble(),
            'fps': fps
        }

class VideoLoader:
    def __init__(self, video_path: str):
        """初始化视频加载器"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = min(int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)  # 限制最大帧数
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 添加帧缓存
        self._frame_cache = {}
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """获取指定帧的图像"""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise ValueError(f"Frame index {frame_idx} out of range [0, {self.total_frames-1}]")
        
        # 检查缓存
        if frame_idx in self._frame_cache:
            return self._frame_cache[frame_idx]
        
        # 定位到指定帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx}")
        
        # 转换为RGB格式并调整大小
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 更新缓存
        self._frame_cache[frame_idx] = frame
        
        # 限制缓存大小
        if len(self._frame_cache) > 1000:  # 最多缓存1000帧
            oldest_key = min(self._frame_cache.keys())
            del self._frame_cache[oldest_key]
        
        return frame
    
    def get_frames_batch(self, start_idx: int, num_frames: int) -> List[np.ndarray]:
        """批量获取连续帧"""
        frames = []
        for i in range(num_frames):
            frame_idx = start_idx + i
            if frame_idx >= self.total_frames:
                break
            frames.append(self.get_frame(frame_idx))
        return frames
    
    def __del__(self):
        """释放视频资源"""
        if hasattr(self, 'cap'):
            self.cap.release()
            self._frame_cache.clear()

class MotionDataset:
    def __init__(self, data_dir: str, input_seconds: int = 3, predict_seconds: int = 2, 
                 fps: int = 10, train_ratio: float = 0.8, random_seed: int = 42):
        """
        初始化动作数据集
        
        Args:
            data_dir: 数据目录路径
            input_seconds: 输入序列长度（秒）
            predict_seconds: 预测序列长度（秒）
            fps: 视频和动作数据的帧率
            train_ratio: 训练集比例
            random_seed: 随机种子
        """
        self.data_dir = Path(data_dir)
        self.input_seconds = input_seconds
        self.predict_seconds = predict_seconds
        self.fps = fps
        self.train_ratio = train_ratio
        self.max_frames = MAX_FRAMES  # 使用全局最大帧数限制
        random.seed(random_seed)
        
        # 计算帧数
        self.input_frames = input_seconds * fps
        self.predict_frames = predict_seconds * fps
        self.total_frames = self.input_frames + self.predict_frames
        self.downsample_factor = int(30 / self.fps)
        
        log_debug(f"\n[DataLoader] Frame configuration:")
        log_debug(f"[DataLoader] Input frames: {self.input_frames} ({input_seconds}s at {fps}fps)")
        log_debug(f"[DataLoader] Predict frames: {self.predict_frames} ({predict_seconds}s at {fps}fps)")
        log_debug(f"[DataLoader] Total frames: {self.total_frames} ({(input_seconds + predict_seconds)}s at {fps}fps)")
        log_debug(f"[DataLoader] Downsample factor: {self.downsample_factor} (from 30fps to {fps}fps)")
        log_debug(f"[DataLoader] Max frames limit: {self.max_frames}")
        
        # 创建处理后的数据目录
        self.processed_data_dir = self.data_dir / "processed_data"
        self.processed_data_dir.mkdir(exist_ok=True)
        
        # 定义要保留的关节
        self.keep_joints = {
            'RootNode', 'Hips', 'RightUpLeg', 'RightLeg', 'RightFoot', 'RightFoot_End', 
            'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftFoot_End', 'Spine', 'Spine1', 
            'Spine2', 'Neck', 'Neck1', 'Head', 'Head_End', 'RightShoulder', 'RightArm', 
            'RightForeArm', 'RightHand', 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand' 
        }
        
        # 添加数据缓存
        self._video_cache = {}
        self._motion_cache = {}
        self._fbx_cache = {}
        self._processed_data_cache = {}
        
        # 获取所有配对的FBX、MP4和CSV文件
        self.data_pairs = self._get_data_pairs()
        
        # 构建数据集
        self.train_data, self.test_data = self._build_dataset()
        
        log_debug(f"\n[DataLoader] Dataset Statistics:")
        log_debug(f"[DataLoader] Total sequences: {len(self.train_data) + len(self.test_data)}")
        log_debug(f"[DataLoader] Training sequences: {len(self.train_data)}")
        log_debug(f"[DataLoader] Testing sequences: {len(self.test_data)}")
        log_debug(f"[DataLoader] Number of joints: {len(self.keep_joints)}")
        
        # 预加载所有FBX数据
        self._preload_fbx_data()
    
    def _get_data_pairs(self) -> List[Dict]:
        """获取所有配对的FBX、MP4和CSV文件"""
        data_pairs = []
        log_debug("\n[DataLoader] Searching for data files...")
        
        # 检查数据目录是否存在
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # 列出目录中的所有文件
        all_files = list(self.data_dir.glob("*"))
        log_debug(f"[DataLoader] Found {len(all_files)} files in directory")
        
        for fbx_file in self.data_dir.glob("*.fbx"):
            log_debug(f"\n[DataLoader] Processing {fbx_file.name}")
            mp4_file = fbx_file.with_suffix('.mp4')
            csv_file = fbx_file.with_suffix('.csv')
            
            # 检查文件是否存在
            if not mp4_file.exists():
                print(f"[DataLoader] Warning: No matching MP4 file found for {fbx_file.name}")
                continue
            if not csv_file.exists():
                print(f"[DataLoader] Warning: No matching CSV file found for {fbx_file.name}")
                continue
            
            log_debug(f"[DataLoader] Found matching files:")
            log_debug(f"[DataLoader] FBX: {fbx_file}")
            log_debug(f"[DataLoader] MP4: {mp4_file}")
            log_debug(f"[DataLoader] CSV: {csv_file}")
            
            # 获取时间戳信息
            log_debug("[DataLoader] Getting video timestamp...")
            video_timestamp, video_frame, video_fps = get_frame_timestamp(str(mp4_file))
            if video_timestamp is None:
                print(f"[DataLoader] Warning: Could not get video timestamp for {mp4_file}")
                continue
            log_debug(f"[DataLoader] Video timestamp: {video_timestamp} at frame {video_frame}")
            
            log_debug("[DataLoader] Getting CSV timestamp...")
            csv_timestamp = get_csv_timestamp(str(csv_file))
            if csv_timestamp is None:
                print(f"[DataLoader] Warning: Could not get CSV timestamp for {csv_file}")
                continue
            log_debug(f"[DataLoader] CSV timestamp: {csv_timestamp}")
            
            # 计算帧偏移
            frame_offset = calculate_frame_offset(video_timestamp, csv_timestamp, video_fps) \
                            + BASE_FBX_FRAME_OFFSET
            log_debug(f"[DataLoader] Calculated frame offset: {frame_offset}")
            
            data_pairs.append({
                'fbx_path': str(fbx_file),
                'mp4_path': str(mp4_file),
                'csv_path': str(csv_file),
                'frame_offset': frame_offset,
                'video_timestamp': video_timestamp,
                'csv_timestamp': csv_timestamp
            })
            log_debug(f"[DataLoader] Successfully added data pair for {fbx_file.name}")
            print(data_pairs)
        
        log_debug(f"\n[DataLoader] Found {len(data_pairs)} valid data pairs")
        return data_pairs
    
    def _build_dataset(self) -> Tuple[List[Dict], List[Dict]]:
        """构建训练集和测试集"""
        all_sequences = []
        log_debug("\n[DataLoader] Building dataset...")
        
        for data_pair in self.data_pairs:
            log_debug(f"\n[DataLoader] Processing data pair:")
            log_debug(f"[DataLoader] FBX: {data_pair['fbx_path']}")
            log_debug(f"[DataLoader] MP4: {data_pair['mp4_path']}")
            
            fbx_path = data_pair['fbx_path']
            mp4_path = data_pair['mp4_path']
            
            try:
                # 加载FBX数据
                log_debug("[DataLoader] Loading FBX data...")
                fbx_animator = FBXAnimator(fbx_path)
                fbx_fps = 30.0  # FBX默认30fps
                log_debug(f"[DataLoader] FBX total frames: {fbx_animator.total_frames}")
                
                # 加载视频数据
                log_debug("[DataLoader] Loading video data...")
                video_loader = VideoLoader(mp4_path)
                log_debug(f"[DataLoader] Video total frames: {video_loader.total_frames}")
                log_debug(f"[DataLoader] Video FPS: {video_loader.fps}")
                
                # 获取时间戳和帧偏移
                video_timestamp, video_frame, video_fps = get_frame_timestamp(mp4_path)
                csv_timestamp = get_csv_timestamp(data_pair['csv_path'])
                frame_offset = calculate_frame_offset(video_timestamp, csv_timestamp, video_fps, fbx_fps) \
                                + BASE_FBX_FRAME_OFFSET
                
                # 计算视频起始帧（跳过前SKIP_VIDEO_FRAMES_RATIO的帧）
                video_start_frame = int(video_loader.total_frames * SKIP_VIDEO_FRAMES_RATIO)
                
                # 计算降采样因子
                downsample_factor = int(video_fps / self.fps)
                log_debug(f"\n[Downsampling Info]")
                log_debug(f"Original FPS: {video_fps}")
                log_debug(f"Target FPS: {self.fps}")
                log_debug(f"Downsample factor: {downsample_factor}")
                
                # 计算每个序列的总帧数
                total_frames = self.input_frames + self.predict_frames
                log_debug(f"[DataLoader] Required sequence length: {total_frames} frames")
                
                # 计算最大可能的起始帧索引（考虑帧偏移）
                max_start_frame = min(
                    fbx_animator.total_frames - total_frames - frame_offset,  # FBX帧数限制
                    video_loader.total_frames - total_frames - video_start_frame  # 视频帧数限制
                )
                log_debug(f"[DataLoader] Maximum start frame: {max_start_frame}")
                
                if max_start_frame < 0:
                    print(f"[DataLoader] Warning: File {fbx_path} is too short for the required sequence length")
                    continue
                
                # 从匹配的时间戳位置开始生成序列
                start_frame = 0  # 从0开始，因为frame_offset会在get_sequence中使用
                sequence_count = 0
                while start_frame + total_frames <= max_start_frame:
                    sequence = {
                        'fbx_path': fbx_path,
                        'mp4_path': mp4_path,
                        'start_frame': start_frame,
                        'input_frames': self.input_frames,
                        'predict_frames': self.predict_frames,
                        'fps': self.fps,
                        'frame_offset': frame_offset,
                        'video_start_frame': video_start_frame
                    }
                    all_sequences.append(sequence)
                    sequence_count += 1
                    start_frame += self.input_frames // 2  # 使用50%的重叠
                
                log_debug(f"[DataLoader] Generated {sequence_count} sequences for this data pair")
                
            except Exception as e:
                print(f"[DataLoader] Error processing data pair: {str(e)}")
                continue
        
        if not all_sequences:
            print("\n[DataLoader] Error: No valid sequences could be created")
            print("[DataLoader] Please check the following:")
            print("1. All required files (FBX, MP4, CSV) exist")
            print("2. Files contain valid data")
            print("3. Timestamps can be read correctly")
            print("4. Files are long enough for the required sequence length")
            raise ValueError("No valid sequences could be created from the provided data files")
        
        # 随机打乱并分割数据集
        random.shuffle(all_sequences)
        split_idx = int(len(all_sequences) * self.train_ratio)
        train_data = all_sequences[:split_idx]
        test_data = all_sequences[split_idx:]
        
        log_debug(f"\n[DataLoader] Dataset building completed:")
        log_debug(f"[DataLoader] Total sequences created: {len(all_sequences)}")
        log_debug(f"[DataLoader] Training sequences: {len(train_data)}")
        log_debug(f"[DataLoader] Testing sequences: {len(test_data)}")
        log_debug(f"[DataLoader] Sequence length: {total_frames} frames")
        log_debug(f"[DataLoader] Input frames: {self.input_frames} frames")
        log_debug(f"[DataLoader] Predict frames: {self.predict_frames} frames")
        
        return train_data, test_data
    
    def get_fps(self):
        """获取当前数据集的帧率"""
        return self.fps
    
    def _filter_joints(self, frame_data: Dict) -> Dict:
        """过滤关节，只保留指定的关节"""
        return {joint: pos for joint, pos in frame_data.items() if joint in self.keep_joints}
    
    def _preload_fbx_data(self):
        """预加载所有FBX数据到内存"""
        log_debug("Preloading FBX data...")
        for data_pair in self.data_pairs:
            fbx_path = data_pair['fbx_path']
            if fbx_path not in self._fbx_cache:
                self._fbx_cache[fbx_path] = FBXAnimator(fbx_path)
                # 预加载所有帧的数据，但限制最大帧数
                self._motion_cache[fbx_path] = {}
                total_frames = min(self._fbx_cache[fbx_path].total_frames, self.max_frames)
                if total_frames < self._fbx_cache[fbx_path].total_frames:
                    log_debug(f"[DataLoader] Truncating FBX {fbx_path} from {self._fbx_cache[fbx_path].total_frames} to {total_frames} frames")
                for frame_idx in range(total_frames):
                    frame_data = self._fbx_cache[fbx_path].get_frame(frame_idx)
                    self._motion_cache[fbx_path][frame_idx] = self._filter_joints(frame_data)
    
    def _align_frames(self, video_frames: List[np.ndarray], motion_frames: List[Dict]) -> Tuple[List[np.ndarray], List[Dict]]:
        """对齐视频帧和动作帧"""
        video_len = len(video_frames)
        motion_len = len(motion_frames)
        
        if video_len != motion_len:
            log_debug(f"Frame count mismatch: video={video_len}, motion={motion_len}")
            # 使用较小的长度
            min_len = min(video_len, motion_len)
            video_frames = video_frames[:min_len]
            motion_frames = motion_frames[:min_len]
            log_debug(f"Aligned to {min_len} frames")
        
        return video_frames, motion_frames

    def _get_save_path(self, sequence_info: Dict) -> Path:
        """获取数据保存路径"""
        fbx_path = Path(sequence_info['fbx_path'])
        start_frame = sequence_info['start_frame']
        frame_offset = sequence_info['frame_offset']
        video_start_frame = sequence_info['video_start_frame']
        
        # 将参数信息和序列信息添加到目录名中
        save_dir = self.processed_data_dir / f"processed_data_{fbx_path.stem}_fps{self.fps}_in{self.input_seconds}_pre{self.predict_seconds}_start{start_frame}_offset{frame_offset}_vstart{video_start_frame}"
        
        # 确保父目录存在
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(exist_ok=True)
        
        return save_dir

    def _save_sequence_data(self, save_dir: Path, video_frames: np.ndarray, 
                          input_motion: np.ndarray, predict_motion: np.ndarray, 
                          timestamps: Dict):
        """保存序列数据到文件"""
        # 保存视频帧
        np.save(save_dir / "video_frames.npy", video_frames)
        
        # 保存动作数据
        np.save(save_dir / "input_motion.npy", input_motion)
        np.save(save_dir / "predict_motion.npy", predict_motion)
        
        # 保存时间戳信息
        with open(save_dir / "timestamps.json", 'w') as f:
            json.dump(timestamps, f, indent=4)

    def _load_sequence_data(self, save_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """从文件加载序列数据"""
        # 加载视频帧
        video_frames = np.load(save_dir / "video_frames.npy")
        
        # 加载动作数据
        input_motion = np.load(save_dir / "input_motion.npy")
        predict_motion = np.load(save_dir / "predict_motion.npy")
        
        # 加载时间戳信息
        with open(save_dir / "timestamps.json", 'r') as f:
            timestamps = json.load(f)
        
        return video_frames, input_motion, predict_motion, timestamps

    def get_sequence(self, sequence_info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """获取一个序列的数据"""
        # 获取保存路径
        save_dir = self._get_save_path(sequence_info)
        
        # 检查是否已经处理过
        if (save_dir / "video_frames.npy").exists():
            log_debug(f"[DataLoader] Loading cached sequence from {save_dir}")
            return self._load_sequence_data(save_dir)
        
        fbx_path = sequence_info['fbx_path']
        mp4_path = sequence_info['mp4_path']
        start_frame = sequence_info['start_frame']
        frame_offset = sequence_info['frame_offset']
        video_start_frame = sequence_info['video_start_frame']
        
        # 获取视频总帧数
        video = cv2.VideoCapture(mp4_path)
        total_frames = min(int(video.get(cv2.CAP_PROP_FRAME_COUNT)), self.max_frames)
        video.release()
        
        # 计算实际的视频起始帧
        video_actual_start = video_start_frame + start_frame
        
        # 确保起始帧不会导致超出总帧数
        if video_actual_start + self.total_frames > total_frames:
            video_actual_start = max(0, total_frames - self.total_frames)
        
        # 计算实际的FBX起始帧
        fbx_actual_start = frame_offset + start_frame
        
        # 计算所有需要的帧索引
        all_frame_indices = []
        
        # 计算输入序列的帧索引
        for i in range(self.input_frames):
            # 计算在30fps下的帧索引
            frame_idx = fbx_actual_start + i * self.downsample_factor
            if frame_idx >= self.max_frames:
                break
            all_frame_indices.append(frame_idx)
        
        # 计算预测序列的帧索引
        for i in range(self.predict_frames):
            # 计算在30fps下的帧索引，从输入序列结束的位置开始
            frame_idx = fbx_actual_start + (self.input_frames + i) * self.downsample_factor
            if frame_idx >= self.max_frames:
                break
            all_frame_indices.append(frame_idx)
        
        # 获取视频帧
        video_frames = []
        for frame_idx in all_frame_indices:
            # 将FBX帧索引转换为视频帧索引
            video_frame_idx = video_actual_start + (frame_idx - fbx_actual_start)
            if video_frame_idx >= total_frames:
                break
            
            # 检查缓存
            if mp4_path not in self._video_cache:
                self._video_cache[mp4_path] = {}
            
            if video_frame_idx in self._video_cache[mp4_path]:
                frame = self._video_cache[mp4_path][video_frame_idx]
            else:
                video = cv2.VideoCapture(mp4_path)
                video.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
                ret, frame = video.read()
                video.release()
                if not ret:
                    break
                frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                self._video_cache[mp4_path][video_frame_idx] = frame
            
            video_frames.append(frame)
        
        video_frames = np.array(video_frames)
        
        # 从缓存获取动作数据
        input_motion = []
        input_timestamps = []
        
        # 获取输入序列的动作数据
        for frame_idx in all_frame_indices[:self.input_frames]:
            try:
                frame_data = self._motion_cache[fbx_path][frame_idx]
                input_motion.append(frame_data)
                input_timestamps.append(frame_idx)
            except KeyError:
                if input_motion:
                    input_motion.append(input_motion[-1])
                    input_timestamps.append(frame_idx)
                else:
                    zero_frame = {joint: np.zeros(3) for joint in self.keep_joints}
                    input_motion.append(zero_frame)
                    input_timestamps.append(frame_idx)
        
        predict_motion = []
        predict_timestamps = []
        
        # 获取预测序列的动作数据
        for frame_idx in all_frame_indices[self.input_frames:]:
            try:
                frame_data = self._motion_cache[fbx_path][frame_idx]
                predict_motion.append(frame_data)
                predict_timestamps.append(frame_idx)
            except KeyError:
                if predict_motion:
                    predict_motion.append(predict_motion[-1])
                    predict_timestamps.append(frame_idx)
                else:
                    zero_frame = {joint: np.zeros(3) for joint in self.keep_joints}
                    predict_motion.append(zero_frame)
                    predict_timestamps.append(frame_idx)
        
        # 对齐视频帧和动作帧
        video_frames, input_motion = self._align_frames(video_frames, input_motion)
        
        # 转换为numpy数组
        joint_names = list(input_motion[0].keys())
        num_joints = len(joint_names)
        
        input_motion_array = np.zeros((len(input_motion), num_joints, 3))
        for i, frame_data in enumerate(input_motion):
            for j, joint_name in enumerate(joint_names):
                input_motion_array[i, j] = frame_data[joint_name]
        
        predict_motion_array = np.zeros((len(predict_motion), num_joints, 3))
        for i, frame_data in enumerate(predict_motion):
            for j, joint_name in enumerate(joint_names):
                predict_motion_array[i, j] = frame_data[joint_name]
        
        timestamps = {
            'input': input_timestamps,
            'predict': predict_timestamps,
            'joint_names': joint_names,
            'video_start': video_actual_start,
            'fbx_start': fbx_actual_start
        }
        
        # 保存处理后的数据
        self._save_sequence_data(save_dir, video_frames, input_motion_array, predict_motion_array, timestamps)
        
        return video_frames, input_motion_array, predict_motion_array, timestamps
    
    def get_train_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """获取一个训练批次的数据"""
        batch_sequences = random.sample(self.train_data, min(batch_size, len(self.train_data)))
        batch_videos = []
        batch_input_motions = []
        batch_predict_motions = []
        batch_timestamps = []
        
        for sequence in batch_sequences:
            video_frames, input_motion, predict_motion, timestamps = self.get_sequence(sequence)
            batch_videos.append(video_frames)
            batch_input_motions.append(input_motion)
            batch_predict_motions.append(predict_motion)
            batch_timestamps.append(timestamps)
        
        return (np.array(batch_videos), 
                np.array(batch_input_motions), 
                np.array(batch_predict_motions), 
                batch_timestamps)
    
    def get_test_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """获取一个测试批次的数据"""
        batch_sequences = random.sample(self.test_data, min(batch_size, len(self.test_data)))
        batch_videos = []
        batch_input_motions = []
        batch_predict_motions = []
        batch_timestamps = []
        
        for sequence in batch_sequences:
            video_frames, input_motion, predict_motion, timestamps = self.get_sequence(sequence)
            batch_videos.append(video_frames)
            batch_input_motions.append(input_motion)
            batch_predict_motions.append(predict_motion)
            batch_timestamps.append(timestamps)
        
        return (np.array(batch_videos), 
                np.array(batch_input_motions), 
                np.array(batch_predict_motions), 
                batch_timestamps)
    
    def __del__(self):
        """清理资源"""
        # 清理视频缓存
        self._video_cache.clear()
        self._motion_cache.clear()
        self._fbx_cache.clear()
