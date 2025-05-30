import torch
import torch.nn as nn
from models.visual_encoder import VisualEncoder
from models.motion_encoder import MotionEncoder
from models.predictor_model import MotionPredictor
import os
import numpy as np
from datetime import datetime

DEBUG_LOG = False
DEBUG_TENSOR = False
def save_tensor_stats(tensor, name, save_dir="debug_tensors"):
    """Save tensor statistics to a file."""
    if not DEBUG_TENSOR:
        return
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f"{name}_{timestamp}.txt")
    
    with open(filename, 'w') as f:
        f.write(f"Tensor: {name}\n")
        f.write(f"Shape: {tensor.shape}\n")
        f.write(f"Device: {tensor.device}\n")
        f.write(f"Type: {tensor.dtype}\n")
        
        if torch.isnan(tensor).any():
            f.write("WARNING: Contains NaN values!\n")
        if torch.isinf(tensor).any():
            f.write("WARNING: Contains Inf values!\n")
            
        # Basic statistics
        if not torch.isnan(tensor).all() and not torch.isinf(tensor).all():
            f.write(f"Mean: {tensor.mean().item()}\n")
            f.write(f"Std: {tensor.std().item()}\n")
            f.write(f"Min: {tensor.min().item()}\n")
            f.write(f"Max: {tensor.max().item()}\n")
            
            # Save actual tensor values
            f.write("\nTensor values:\n")
            np.savetxt(f, tensor.detach().cpu().numpy().reshape(-1, tensor.shape[-1]))
            
    return filename

def check_gradients(model, name="model"):
    """Check gradients of model parameters."""
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_stats[name] = {
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'min': grad.min().item(),
                'max': grad.max().item(),
                'has_nan': torch.isnan(grad).any().item(),
                'has_inf': torch.isinf(grad).any().item()
            }
    return grad_stats

class VideoMotionModel(nn.Module):
    def __init__(self, 
                 visual_encoder_args=None, 
                 motion_encoder_args=None,
                 predictor_args=None,
                 device=None):
        super().__init__()
        self.device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if DEBUG_LOG:
            print(f"\n[VideoMotionModel] Initializing with device: {self.device}")
            print(f"[VideoMotionModel] Visual encoder args: {visual_encoder_args}")
            print(f"[VideoMotionModel] Motion encoder args: {motion_encoder_args}")
            print(f"[VideoMotionModel] Predictor args: {predictor_args}")
        
        # Initialize encoders
        self.visual_encoder = VisualEncoder(**(visual_encoder_args or {}), device=self.device)
        self.motion_encoder = MotionEncoder(**(motion_encoder_args or {})).to(self.device)
        
        # Initialize predictor
        predictor_args = predictor_args or {}
        self.predictor = MotionPredictor(
            input_dim=512,  # 256(视觉) + 256(动作)
            hidden_dim=predictor_args.get('hidden_dim', 512),
            output_dim=predictor_args.get('output_dim', 75),  # 25 joints * 3 coordinates
            num_layers=predictor_args.get('num_layers', 2),
            dropout=predictor_args.get('dropout', 0.1),
            predict_frames=predictor_args.get('predict_frames', 20)
        ).to(self.device)

    def forward(self, video_frames, motion_seq):
        """
        Args:
            video_frames: (batch, num_frames, 3, H, W)
            motion_seq: (batch, seq_len, motion_dim)
        Returns:
            predicted_motion: (batch, predict_frames, output_dim)
        """
        if DEBUG_LOG:
            print(f"\n[VideoMotionModel] Input shapes:")
            print(f"[VideoMotionModel] video_frames: {video_frames.shape}")
            print(f"[VideoMotionModel] motion_seq: {motion_seq.shape}")
            
        # Save input statistics
        save_tensor_stats(video_frames, "input_video_frames")
        save_tensor_stats(motion_seq, "input_motion_seq")
        
        # Get features from encoders
        # visual_feat = self.visual_encoder(video_frames)  # [batch_size, 1568, 768]
        # 临时使用全0张量替代视觉特征
        visual_feat = torch.zeros((video_frames.size(0), 768), device=video_frames.device)
        motion_feat, _ = self.motion_encoder(motion_seq)  # [batch_size, 30, 256]

        if DEBUG_LOG:
            print(f"[VideoMotionModel] visual_feat shape: {visual_feat.shape}")
            print(f"[VideoMotionModel] motion_feat shape: {motion_feat.shape}")
        save_tensor_stats(visual_feat, "visual_features")
        save_tensor_stats(motion_feat, "motion_features")
        
        # Ensure batch sizes match
        if visual_feat.size(0) != motion_feat.size(0):
            # If visual features have batch size 1, expand to match motion features
            if visual_feat.size(0) == 1:
                visual_feat = visual_feat.expand(motion_feat.size(0), -1, -1)
            else:
                raise ValueError(f"Batch size mismatch: visual_feat {visual_feat.size(0)} != motion_feat {motion_feat.size(0)}")
        
        # Predict future motion
        predicted_motion = self.predictor(
            visual_features=visual_feat,
            motion_features=motion_feat,
            initial_motion=motion_seq
        )
        # Save predicted motion statistics
        save_tensor_stats(predicted_motion, "predicted_motion")
        
        if DEBUG_LOG:
            print(f"[VideoMotionModel] predicted_motion shape: {predicted_motion.shape}")
            print(f"[VideoMotionModel] Predicted Motion Min: {predicted_motion.min().item()}, Max: {predicted_motion.max().item()}")
        
        return predicted_motion

    def train_step(self, video_frames, input_motion, target_motion, optimizer, loss_fn):
        # 前向传播
        predicted_motion = self.forward(video_frames, input_motion)
        
        # 计算损失
        losses = loss_fn(predicted_motion, target_motion)
        total_loss = losses['total_loss']
        
        # 检查损失值
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if DEBUG_LOG:
                print("\n[DEBUG] NaN/Inf detected in loss!")
                print(f"Total loss: {total_loss.item()}")
                print("Checking gradients before backward pass...")
                grad_stats = check_gradients(self)
                for name, stats in grad_stats.items():
                    print(f"\nGradient stats for {name}:")
                    for k, v in stats.items():
                        print(f"{k}: {v}")
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        
        # 检查梯度
        grad_stats = check_gradients(self)
        if any(stats['has_nan'] or stats['has_inf'] for stats in grad_stats.values()):
            print("\n[DEBUG] NaN/Inf detected in gradients after backward pass!")
            for name, stats in grad_stats.items():
                if stats['has_nan'] or stats['has_inf']:
                    print(f"\nGradient stats for {name}:")
                    for k, v in stats.items():
                        print(f"{k}: {v}")
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        # 更新权重
        optimizer.step()
        
        return losses
