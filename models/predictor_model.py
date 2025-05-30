# models/motion_predictor_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(256, dim)  # 添加投影层处理动作特征
        
    def forward(self, query, key, value):
        # 投影动作特征到正确的维度
        key = self.projection(key)  # [batch_size, seq_len, dim]
        value = self.projection(value)  # [batch_size, seq_len, dim]
        
        attn_output, _ = self.attention(query, key, value)
        return self.norm(query + attn_output)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(768, 512)  # 处理视觉特征
        self.fc2 = nn.Linear(256, 512)  # 处理动作特征
        self.fc3 = nn.Linear(1024, 1024)  # 处理融合特征
        self.gate = nn.Sigmoid()
        
    def forward(self, x1, x2):
        # 将两个特征投影到相同维度
        x1_proj = self.fc1(x1)  # [batch_size, seq_len, 512]
        x2_proj = self.fc2(x2)  # [batch_size, seq_len, 512]
        
        # 连接投影后的特征
        combined = torch.cat([x1_proj, x2_proj], dim=-1)  # [batch_size, seq_len, 1024]
        
        # 计算门控权重
        gates = self.gate(self.fc3(combined))  # [batch_size, seq_len, 1024]
        
        # 将门控分成两部分
        g1, g2 = gates.chunk(2, dim=-1)  # 每个都是 [batch_size, seq_len, 512]
        
        # 应用门控并连接
        fused = torch.cat([g1 * x1_proj, g2 * x2_proj], dim=-1)  # [batch_size, seq_len, 1024]
        
        return fused


class MotionPredictor(nn.Module):
    """Motion predictor that fuses visual and motion features to predict future motion."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 predict_frames: int = 20):
        """
        动作预测器
        
        Args:
            input_dim: 输入维度 (visual_feature_dim + motion_feature_dim)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度 (num_joints * 3)
            num_layers: LSTM层数
            dropout: Dropout比率
            predict_frames: 预测帧数
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.predict_frames = predict_frames
        
        # 特征处理层 - 只处理动作特征，视觉特征保持原始维度
        self.motion_projection = nn.Linear(256, 256)
        
        # 时序特征处理
        self.visual_temporal = nn.LSTM(
            input_size=768,  # 使用原始视觉特征维度
            hidden_size=768,  # 保持维度一致
            num_layers=1,
            batch_first=True
        )
        
        # 特征融合层 - 修改输入维度为 768 + 256
        self.fusion = GatedFusion(1024)  # 768 + 256
        
        # 交叉注意力层 - 修改维度
        self.cross_attention = CrossAttention(1024)  # 768 + 256
        
        # LSTM层 - 修改输入维度
        self.lstm = nn.LSTM(
            input_size=1024,  # 768 + 256
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 时间注意力层
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=8, 
            batch_first=True
        )
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 残差连接 - 修改输入维度
        self.residual_projection = nn.Linear(1024, hidden_dim)  # 768 + 256
        
        # 动态生成层
        self.dynamic_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0.0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, 
                visual_features: torch.Tensor,
                motion_features: torch.Tensor,
                initial_motion: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            visual_features: 视觉特征 (batch_size, 768) 或 (batch_size, num_patches, 768)
            motion_features: 动作特征 (batch_size, 30, 256)
            initial_motion: 初始动作序列 (batch_size, input_frames, motion_feature_dim)
            
        Returns:
            predicted_motion: 预测的动作序列 (batch_size, predict_frames, motion_feature_dim)
        """
        batch_size = visual_features.size(0)
        
        # 处理视觉特征 - 保持原始维度
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(1)  # [batch_size, 1, 768]
        else:
            # 使用LSTM处理时序信息
            visual_features, _ = self.visual_temporal(visual_features)  # [batch_size, seq_len, 768]
        
        # 处理动作特征
        motion_features = self.motion_projection(motion_features)  # [batch_size, seq_len, 256]
        
        # 特征融合
        fused_features = self.fusion(
            visual_features.expand(-1, motion_features.size(1), -1),
            motion_features
        )  # [batch_size, seq_len, 1024]
        
        # 交叉注意力 - 使用融合特征作为query，动作特征作为key和value
        attended_features = self.cross_attention(fused_features, motion_features, motion_features)
        
        # 生成时序特征
        visual_features = visual_features.expand(batch_size, self.predict_frames, -1)  # [batch_size, predict_frames, 768]
        motion_features = motion_features.expand(batch_size, self.predict_frames, -1)  # [batch_size, predict_frames, 256]
        
        # 合并特征
        combined_features = torch.cat([visual_features, motion_features], dim=-1)  # [batch_size, predict_frames, 1024]
        
        # 初始化LSTM状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(visual_features.device)
        
        # LSTM前向传播
        lstm_out, _ = self.lstm(combined_features, (h0, c0))
        
        # 时间注意力
        temporal_out, _ = self.temporal_attention(lstm_out, lstm_out, lstm_out)
        
        # 残差连接
        residual = self.residual_projection(attended_features.mean(dim=1, keepdim=True))
        residual = residual.expand(-1, self.predict_frames, -1)
        
        # 生成动态特征
        dynamic_features = self.dynamic_generator(temporal_out)
        
        # 最终输出
        final_features = temporal_out + residual
        predicted_motion = self.output_projection(final_features) + dynamic_features
        
        return predicted_motion

def test_motion_predictor():
    """Test the motion prediction model functionality."""
    batch_size = 4
    seq_len = 15
    visual_feat_dim = 768
    motion_feat_dim = 256
    output_dim = 75
    
    # Create sample inputs
    visual_feat = torch.randn(batch_size, visual_feat_dim)  # 2D visual features
    motion_feat = torch.randn(batch_size, seq_len, motion_feat_dim)
    
    # Initialize predictor
    predictor = MotionPredictor(
        input_dim=visual_feat_dim + motion_feat_dim,
        hidden_dim=512,
        output_dim=output_dim,
        num_layers=2,
        dropout=0.1,
        predict_frames=20
    )
    
    # Forward pass
    output = predictor(visual_feat, motion_feat, torch.randn(batch_size, seq_len, motion_feat_dim))
    
    # Print shapes
    print(f"Visual feature shape: {visual_feat.shape}")
    print(f"Motion feature shape: {motion_feat.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == '__main__':
    test_motion_predictor()