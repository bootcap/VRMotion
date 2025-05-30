# models/motion_encoder.py
import torch
import torch.nn as nn
import math
import numpy as np

# 全局调试开关
DEBUG = False

def log_debug(*args, **kwargs):
    """调试日志函数，只有在DEBUG=True时才会输出"""
    if DEBUG:
        print(*args, **kwargs)
        
class MotionEncoder(nn.Module):
    """LSTM-based motion encoder for processing motion capture data."""
    
    def __init__(self, 
                 input_dim, 
                 embed_dim=256, 
                 hidden_dim=512, 
                 num_layers=2,
                 dropout=0.1):
        """
        Initialize the motion encoder.
        
        Args:
            input_dim: Input dimension (num_joints * 3)
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate for LSTM layers
        """
        super(MotionEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        log_debug(f"[MotionEncoder] Initialized with input_dim={input_dim}, embed_dim={embed_dim}, "
              f"hidden_dim={hidden_dim}, num_layers={num_layers}")
        
        # Input embedding layer
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim, embed_dim)
        
    def forward(self, x):
        """
        Forward pass through the motion encoder.
        
        Args:
            x: Tensor of shape [batch_size, seq_len, num_joints * 3]
               representing motion capture data
               
        Returns:
            output: Tensor of shape [batch_size, seq_len, embed_dim]
            hidden: Tuple of (h_n, c_n) hidden states from LSTM
        """
        batch_size, seq_len, _ = x.shape
        
        log_debug(f"\n[MotionEncoder] Input tensor shape: {x.shape}")
        log_debug(f"[MotionEncoder] Expected input dimension: {self.input_dim}")
        log_debug(f"[MotionEncoder] Actual input dimension: {x.size(-1)}")
        log_debug(f"[MotionEncoder] Number of joints (inferred): {x.size(-1) // 3}")
        
        # Validate input dimensions
        if x.size(-1) != self.input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {self.input_dim}, got {x.size(-1)}. "
                           f"Please ensure input_dim matches the flattened joint positions (num_joints * 3)")
        
        # Input embedding
        x_embed = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        log_debug(f"[MotionEncoder] After embedding shape: {x_embed.shape}")
        
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x_embed)  # lstm_out: [batch_size, seq_len, hidden_dim]
        log_debug(f"[MotionEncoder] After LSTM shape: {lstm_out.shape}")
        
        # Project to output dimension
        output = self.projection(lstm_out)  # [batch_size, seq_len, embed_dim]
        log_debug(f"[MotionEncoder] Final output shape: {output.shape}")
        
        return output, hidden

def test_motion_encoder():
    """Test the motion encoder functionality."""
    # Create sample inputs
    batch_size = 4
    num_frames = 30
    motion_dim = 75  # 36 joints * 3 coordinates
    
    past_motion = torch.randn(batch_size, num_frames, motion_dim)
    
    # Create model
    motion_encoder = MotionEncoder(input_dim=motion_dim)
    
    # Forward pass
    motion_features, _ = motion_encoder(past_motion)
    
    log_debug(f"Past motion shape: {past_motion.shape}")
    log_debug(f"Motion features shape: {motion_features.shape}")

if __name__ == '__main__':
    test_motion_encoder()