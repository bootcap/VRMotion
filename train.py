# training/train.py
import os
import sys
import torch
torch.cuda.set_device(torch.device(f"cuda:0"))
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from utils.data_loader import MotionDataset
from utils.losses import MotionLoss
from utils.scheduler import TeacherForcingScheduler, get_learning_rate_scheduler
from models.model import VideoMotionModel
from utils.visualization_utils import visualize_batch_results
from pathlib import Path
from tqdm import tqdm
import random
from utils.visualization_utils import MotionVisualizer

# --- Configuration ---
CONFIG = {
    "experiment_name": "vr_motion_pred_v1",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,

    # Data
    "data_dir": "data",
    "input_seconds": 2,  # 输入序列长度（秒）
    "predict_seconds": 2,  # 预测序列长度（秒）
    "fps": 8,  # 视频和动作数据的帧率
    "batch_size": 40,
    "train_ratio": 0.8,  # 训练集比例

    # Model: Visual Encoder
    "visual_encoder_args": {
        'image_size': 224,
        'patch_size': 16,
        'num_frames': 16,  # input_frames (3s * 10fps)
        'embed_dim': 768,
        'pretrained_model_name': "OpenGVLab/VideoMAEv2-Base"
    },

    # Model: Motion Encoder
    "motion_feature_dim": 75,  # 25 joints * 3 coords
    "motion_encoder_args": {
        'input_dim': 75,  # 25 joints * 3 coords
        'embed_dim': 256,
        'hidden_dim': 512,
        'num_layers': 2,
        'dropout': 0.1
    },

    # Model: Predictor
    "predictor_args": {
        'hidden_dim': 512,
        'output_dim': 75,  # 25 joints * 3 coords
        'num_layers': 2,
        'dropout': 0.1,
        'predict_frames': 16  # predict_seconds * fps = 2 * 8
    },

    # Training
    "epochs": 100,
    "learning_rate": 1e-4,
    "optimizer": "adamw",
    "weight_decay": 1e-5,
    "loss_type": "mse",
    "velocity_loss_weight": 0.05,
    "acceleration_loss_weight": 0.0,

    # Schedulers
    "lr_scheduler_type": "reduce_on_plateau",
    "lr_step_size": 20,
    "lr_gamma": 0.5,
    "lr_patience": 5,
    "teacher_forcing_initial_ratio": 1.0,
    "teacher_forcing_final_ratio": 0.0,
    "teacher_forcing_decay_epochs": 50,
    "teacher_forcing_decay_type": "linear",

    # Checkpointing & Logging
    "checkpoint_dir": "./checkpoints",
    "log_interval": 10,
    "save_best_only": True,
}

# --- Helper Functions ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def save_checkpoint(state, is_best, checkpoint_dir, filename="checkpoint.pth.tar", best_filename="model_best.pth.tar"):
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_filepath)
        print(f" => Saved new best model to {best_filepath}")

def load_checkpoint(checkpoint_path, model, optimizer=None, lr_scheduler=None):
    if not os.path.isfile(checkpoint_path):
        print(f" => No checkpoint found at '{checkpoint_path}'")
        return None, 0, float('inf')
    
    print(f" => Loading checkpoint '{checkpoint_path}'")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    
    state_dict = checkpoint['state_dict']
    if any(key.startswith('module.') for key in state_dict.keys()):
        print("Adjusting state_dict keys from DataParallel/DistributedDataParallel model.")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    start_epoch = checkpoint.get('epoch', 0)
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    if optimizer and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")

    if lr_scheduler and 'lr_scheduler' in checkpoint:
        try:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        except Exception as e:
            print(f"Warning: Could not load LR scheduler state: {e}")
            
    print(f" => Loaded checkpoint '{checkpoint_path}' (epoch {start_epoch}, best_val_loss {best_val_loss:.4f})")
    return model, start_epoch, best_val_loss

# --- Main Training Function ---
def main_train(config):
    set_seed(config["seed"])
    device = torch.device(config["device"])

    # Create checkpoint directory
    checkpoint_dir = os.path.join(config["checkpoint_dir"], config["experiment_name"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(checkpoint_dir, "config.json"), 'w') as f:
        json.dump(config, f, indent=4)

    # --- 1. Data ---
    print("Loading data...")
    dataset = MotionDataset(
        data_dir=config["data_dir"],
        input_seconds=config["input_seconds"],
        predict_seconds=config["predict_seconds"],
        fps=config["fps"],
        train_ratio=config["train_ratio"]
    )
    print(f"Dataset loaded. Train sequences: {len(dataset.train_data)}, Test sequences: {len(dataset.test_data)}")

    # --- 2. Model ---
    print("Initializing model...")
    model = VideoMotionModel(
        visual_encoder_args={"pretrained_model_name": "OpenGVLab/VideoMAEv2-Base"},
        motion_encoder_args=config["motion_encoder_args"],
        predictor_args=config["predictor_args"],
        device=device
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {num_params/1e6:.2f}M trainable parameters.")

    # --- Test forward ---
    model.eval()
    with torch.no_grad():
        video_frames, input_motion, predict_motion, timestamps = dataset.get_train_batch(config["batch_size"])
        # video_frames: (batch, num_frames, H, W, 3) -> (batch, num_frames, 3, H, W)
        video_frames = torch.from_numpy(video_frames).float().permute(0, 1, 4, 2, 3).to(device)
        # input_motion: (batch, seq_len, joints, 3) -> (batch, seq_len, joints*3)
        b, t, j, c = input_motion.shape
        input_motion = torch.from_numpy(input_motion).float().reshape(b, t, j * c).to(device)
        predicted_motion = model(video_frames, input_motion)
        print(f"Predicted motion shape: {predicted_motion.shape}")

    # --- 3. Optimizer ---
    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
    print(f"Optimizer: {config['optimizer']}")

    # --- 4. Loss Function ---
    criterion = MotionLoss(
        position_weight=1.0,
        velocity_weight=0.5,
        acceleration_weight=0.3
    ).to(device)
    print(f"Loss: position_w: 1.0, velocity_w: 0.5, acceleration_w: 0.3")

    # --- 5. Schedulers ---
    lr_scheduler = None
    if config["lr_scheduler_type"]:
        lr_scheduler = get_learning_rate_scheduler(
            optimizer, 
            scheduler_type=config["lr_scheduler_type"],
            step_size=config["lr_step_size"],
            gamma=config["lr_gamma"],
            patience=config["lr_patience"]
        )
        print(f"LR Scheduler: {config['lr_scheduler_type']}")

    tf_scheduler = TeacherForcingScheduler(
        initial_ratio=config["teacher_forcing_initial_ratio"],
        final_ratio=config["teacher_forcing_final_ratio"],
        decay_epochs=config["teacher_forcing_decay_epochs"],
        decay_type=config["teacher_forcing_decay_type"]
    )
    print(f"TF Scheduler: initial={tf_scheduler.initial_ratio}, final={tf_scheduler.final_ratio}, decay_epochs={tf_scheduler.decay_epochs}")

    # --- Load Checkpoint (if any) ---
    start_epoch = 0
    best_val_loss = float('inf')
    resume_checkpoint_path = None  # Set to a path to resume
    if resume_checkpoint_path and os.path.exists(resume_checkpoint_path):
        model, start_epoch, best_val_loss = load_checkpoint(resume_checkpoint_path, model, optimizer, lr_scheduler)
        for i in range(start_epoch):
            tf_scheduler.step(i)
        print(f"Resuming training from epoch {start_epoch}. Current TF ratio: {tf_scheduler.get_ratio():.4f}")

    # --- Training Loop ---
    print(f"\nStarting training for {config['epochs']} epochs on {device}...")
     
    for epoch in range(start_epoch, config["epochs"]):
        epoch_start_time = time.time()
        
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0
        current_tf_ratio = tf_scheduler.step(epoch)
        
        # Calculate number of batches per epoch
        num_batches = len(dataset.train_data) // config["batch_size"]
        if len(dataset.train_data) % config["batch_size"] != 0:
            num_batches += 1
            
        # Create progress bar for training
        train_pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for batch_idx in train_pbar:
            # Get training batch
            video_frames, input_motion, predict_motion, timestamps = dataset.get_train_batch(config["batch_size"])
            
            # Convert to tensors and move to device
            video_frames = torch.from_numpy(video_frames).float().permute(0, 1, 4, 2, 3).to(device)
            input_motion = torch.from_numpy(input_motion).float().to(device)
            b, t, j, c = input_motion.shape
            input_motion = input_motion.reshape(b, t, j * c)
            predict_motion = torch.from_numpy(predict_motion).float().to(device)
            b, t, j, c = predict_motion.shape
            predict_motion = predict_motion.reshape(b, t, j * c)
            
            optimizer.zero_grad()
            
            # Model forward pass
            predicted_motion = model(video_frames, input_motion)
            predicted_motion = predicted_motion[:, :predict_motion.size(1), :]
            
            losses = criterion(predicted_motion, predict_motion)
            losses['total_loss'].backward()
            optimizer.step()

            train_loss_accum += losses['total_loss'].item()
            
            # Update progress bar with current loss
            train_pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'tf_ratio': f"{current_tf_ratio:.3f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        avg_train_loss = train_loss_accum / num_batches

        # --- Validation Phase ---
        model.eval()
        val_loss_accum = 0.0
        
        # Calculate number of validation batches
        num_val_batches = len(dataset.test_data) // config["batch_size"]
        if len(dataset.test_data) % config["batch_size"] != 0:
            num_val_batches += 1
            
        # Create progress bar for validation
        val_pbar = tqdm(range(num_val_batches), desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
        
        with torch.no_grad():
            for batch_idx in val_pbar:
                # Get validation batch
                video_frames_val, input_motion_val, predict_motion_val, timestamps_val = dataset.get_test_batch(config["batch_size"])
                
                # Convert to tensors and move to device
                video_frames_val = torch.from_numpy(video_frames_val).float().permute(0, 1, 4, 2, 3).to(device)
                input_motion_val = torch.from_numpy(input_motion_val).float().to(device)
                b, t, j, c = input_motion_val.shape
                input_motion_val = input_motion_val.reshape(b, t, j * c)
                predict_motion_val = torch.from_numpy(predict_motion_val).float().to(device)
                b, t, j, c = predict_motion_val.shape
                predict_motion_val = predict_motion_val.reshape(b, t, j * c)

                predicted_motion_val = model(video_frames_val, input_motion_val)
                predicted_motion_val = predicted_motion_val[:, :predict_motion_val.size(1), :]
                
                val_loss = criterion(predicted_motion_val, predict_motion_val)
                val_loss_accum += val_loss['total_loss'].item()
            
                # Update progress bar with current loss
                val_pbar.set_postfix({'loss': f"{val_loss['total_loss'].item():.4f}"})
        
        avg_val_loss = val_loss_accum / num_val_batches
        epoch_duration = time.time() - epoch_start_time

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{config['epochs']} completed in {epoch_duration:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"TF Ratio: {current_tf_ratio:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if epoch % 10 == 0:
            # --- Visualization of Validation Results ---
            print("\nVisualizing validation results...")
            # 创建可视化输出目录
            vis_output_dir = os.path.join(checkpoint_dir, f"visualization_epoch_{epoch+1}")
            os.makedirs(vis_output_dir, exist_ok=True)
            
            # 随机选择5个样本进行可视化
            num_samples = min(3, len(dataset.test_data))
            sample_indices = random.sample(range(len(dataset.test_data)), num_samples)
            
            for sample_idx in sample_indices:
                # 获取单个样本数据
                sequence = dataset.test_data[sample_idx]
                video_frames, input_motion, predict_motion, timestamps = dataset.get_sequence(sequence)
                
                # 转换为tensor并预测
                # 确保视频帧维度正确 (batch, frames, height, width, channels)
                if len(video_frames.shape) == 4:  # 如果是 (frames, height, width, channels)
                    video_frames = np.expand_dims(video_frames, 0)  # 添加batch维度
                video_frames_tensor = torch.from_numpy(video_frames).float().permute(0, 1, 4, 2, 3).to(device)
                
                # 处理动作数据
                input_motion_tensor = torch.from_numpy(input_motion).float().to(device)
                # 检查并调整动作数据维度
                if len(input_motion_tensor.shape) == 3:  # 如果是 (frames, joints, 3)
                    input_motion_tensor = input_motion_tensor.unsqueeze(0)  # 添加batch维度
                b, t, j, c = input_motion_tensor.shape
                input_motion_tensor = input_motion_tensor.reshape(b, t, j * c)
                
                # 模型预测
                with torch.no_grad():
                    predicted_motion = model(video_frames_tensor, input_motion_tensor)
                    predicted_motion = predicted_motion.cpu().numpy()
                    predicted_motion = predicted_motion.reshape(-1, j, c)  # 重塑回原始形状
                
                # 创建样本特定的输出目录
                sample_output_dir = os.path.join(vis_output_dir, f"sample_{sample_idx}")
                os.makedirs(sample_output_dir, exist_ok=True)
                
                # 使用新的可视化函数
                from utils.visualization_utils import visualize_prediction_sequence
                visualize_prediction_sequence(
                    input_motion=input_motion,
                    predicted_motion=predicted_motion,
                    target_motion=predict_motion,
                    save_dir=sample_output_dir,
                    fps=dataset.fps
                )
                print(f"Visualized sample {sample_idx} saved to {sample_output_dir}")

        # --- LR Scheduler Step ---
        if lr_scheduler:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(avg_val_loss)
            else:
                lr_scheduler.step()
        
        # --- Checkpointing ---
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
            print(f" => New best validation loss: {best_val_loss:.4f}")
        
        checkpoint_data = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': config
        }
        if lr_scheduler:
            checkpoint_data['lr_scheduler'] = lr_scheduler.state_dict()

        if config["save_best_only"]:
            if is_best:
                save_checkpoint(checkpoint_data, is_best=True, checkpoint_dir=checkpoint_dir)
        else:
            save_checkpoint(checkpoint_data, is_best=False, checkpoint_dir=checkpoint_dir)
            if is_best:
                save_checkpoint(checkpoint_data, is_best=True, checkpoint_dir=checkpoint_dir)

    print(f"\nTraining finished. Best validation loss: {best_val_loss:.4f}")
    final_model_path = os.path.join(checkpoint_dir, "model_final.pth.tar")
    torch.save({'epoch': config['epochs'], 'state_dict': model.state_dict(), 'config': config}, final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    print("Starting training script...")
    main_train(CONFIG)
