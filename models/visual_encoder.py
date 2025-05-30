# models/visual_encoder.py
import torch
import torch.nn as nn
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
import os
from huggingface_hub import login
import numpy as np

# 全局调试开关
DEBUG = False

def log_debug(*args, **kwargs):
    """调试日志函数，只有在DEBUG=True时才会输出"""
    if DEBUG:
        print(*args, **kwargs)

class VisualEncoder(nn.Module):
    """Visual encoder using HuggingFace VideoMAEv2 Base (for feature extraction)."""
    def __init__(self, pretrained_model_name='OpenGVLab/VideoMAEv2-Base', device=None, token=None):
        super().__init__()
        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-Base", trust_remote_code=True)
        self.model = AutoModel.from_pretrained('OpenGVLab/VideoMAEv2-Base', config=config, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.processor = None
        self.max_frames = 16  # VideoMAE 标准输入帧数

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for video feature extraction.
        
        Args:
            video: Tensor of shape (batch_size, num_frames, 3, H, W)
            
        Returns:
            Tensor of shape (batch_size, num_patches, hidden_size)
        """
        log_debug(f"[VisualEncoder] input video shape: {video.shape}, dtype: {video.dtype}, max: {video.max()}, min: {video.min()}")
        
        # 1. Input validation
        if video.ndim != 5 or video.shape[2] != 3:
            raise ValueError(f"Input video must be (batch, num_frames, 3, H, W), got {video.shape}")
        
        # 2. Normalize to [0, 1] range if needed
        if video.max() > 1.1:
            video = video / 255.0
        
        # 3. Truncate frames if necessary
        if video.shape[1] > self.max_frames:
            log_debug(f"[VisualEncoder] Truncating video from {video.shape[1]} to {self.max_frames} frames")
            # 均匀采样帧
            indices = torch.linspace(0, video.shape[1]-1, self.max_frames).long()
            video = video[:, indices]
        
        # 4. Prepare input for model
        # (B, T, C, H, W) -> (B, C, T, H, W)
        inputs = {
            'pixel_values': video.permute(0, 2, 1, 3, 4)
        }
        log_debug(f"[VisualEncoder] model input shape: {inputs['pixel_values'].shape}")
        
        # 5. Forward pass through model
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 6. Extract features
        feats = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs
        log_debug(f"[VisualEncoder] output feats shape: {feats.shape if hasattr(feats, 'shape') else type(feats)}")
        
        return feats

def test_visual_encoder():
    """Test the visual encoder functionality."""
    # Create sample input
    batch_size = 4
    num_frames = 30  # 测试超过16帧的情况
    channels = 3
    height = 224
    width = 224
    
    # 创建随机输入张量
    x = torch.randn(batch_size, num_frames, channels, height, width)
    
    # Create model
    model = VisualEncoder()
    
    # Forward pass
    output = model(x)
    
    log_debug(f"Input shape: {x.shape}")
    log_debug(f"Output shape: {output.shape}")
    
    # Test with different batch sizes
    x = torch.randn(1, num_frames, channels, height, width)
    output = model(x)
    log_debug(f"Single sample output shape: {output.shape}")

if __name__ == '__main__':
    test_visual_encoder()