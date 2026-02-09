"""
Sign language recognition model architecture.

Multi-branch LSTM model with hand, body, and face modalities.
This implementation matches the training architecture exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Literal


class HandBranch(nn.Module):
    """Branch for processing hand keypoints."""

    def __init__(
        self,
        input_dim: int = 21 * 4 * 2,  # 2 hands * 21 keypoints * 4 values = 168
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection with ReLU (matches training)
        x = self.input_proj(x)
        x = F.relu(x)

        # LSTM
        x, _ = self.lstm(x)

        # Layer normalization
        x = self.layer_norm(x)

        return x


class BodyBranch(nn.Module):
    """Branch for processing body/pose keypoints."""

    def __init__(
        self,
        input_dim: int = 33 * 4,  # 33 keypoints * 4 values = 132
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection with ReLU (matches training)
        x = self.input_proj(x)
        x = F.relu(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class FaceBranch(nn.Module):
    """Branch for processing face keypoints."""

    def __init__(
        self,
        input_dim: int = 468 * 4,  # 468 keypoints * 4 values = 1872
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Dimensionality reduction (468 landmarks is a lot)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        return x


class MultimodalEncoder(nn.Module):
    """
    Multimodal encoder that combines three branches (hands, body, face).
    
    This matches the training architecture exactly.
    """

    def __init__(
        self,
        hand_input_dim: int = 21 * 4 * 2,  # 2 hands * 21 keypoints * 4 values = 168
        body_input_dim: int = 33 * 4,       # 33 keypoints * 4 values = 132
        face_input_dim: int = 468 * 4,      # 468 keypoints * 4 values = 1872
        hidden_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Three branches
        self.hand_branch = HandBranch(hand_input_dim, hidden_dim, num_layers, dropout)
        self.body_branch = BodyBranch(body_input_dim, hidden_dim, num_layers, dropout)
        self.face_branch = FaceBranch(face_input_dim, hidden_dim, num_layers, dropout)

        # Fusion of embeddings
        # Each branch produces hidden_dim, we combine all 3
        fusion_dim = hidden_dim * 3
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def forward(
        self,
        hand_keypoints: torch.Tensor,
        body_keypoints: torch.Tensor,
        face_keypoints: torch.Tensor
    ) -> torch.Tensor:
        """
        Process keypoints from all three branches and fuse them.

        Args:
            hand_keypoints: Tensor (batch, seq_len, hand_input_dim)
            body_keypoints: Tensor (batch, seq_len, body_input_dim)
            face_keypoints: Tensor (batch, seq_len, face_input_dim)

        Returns:
            Fused tensor (batch, seq_len, output_dim)
        """
        # Process each branch
        hand_emb = self.hand_branch(hand_keypoints)
        body_emb = self.body_branch(body_keypoints)
        face_emb = self.face_branch(face_keypoints)

        # Concatenate embeddings
        fused = torch.cat([hand_emb, body_emb, face_emb], dim=-1)

        # Final projection
        output = self.fusion(fused)

        return output


class SignLanguageClassifier(nn.Module):
    """
    Classification model that wraps MultimodalEncoder.
    
    Takes multimodal keypoint sequences and produces class logits.
    Handles temporal pooling to convert sequence embeddings [B, T, D]
    to fixed-size representations [B, D] for classification.
    
    This matches the training architecture exactly.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        pooling: Literal["mean", "max", "last"] = "mean",
        dropout: float = 0.1
    ):
        super().__init__()
        
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        
        self.encoder = encoder
        self.pooling = pooling
        self.num_classes = num_classes
        
        # Get encoder output dimension
        encoder_dim = getattr(encoder, 'output_dim', 512)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder_dim, num_classes)
    
    def forward(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: keypoints → embeddings → pooled → logits.
        
        Args:
            hand: Hand keypoints [B, T, hand_dim]
            body: Body keypoints [B, T, body_dim]
            face: Face keypoints [B, T, face_dim]
            lengths: Original sequence lengths [B] (for masked pooling)
            mask: Boolean mask [B, T] where True = valid position
        
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Encode: [B, T, D]
        embeddings = self.encoder(hand, body, face)
        
        # Pool: [B, T, D] → [B, D]
        pooled = self._temporal_pool(embeddings, lengths, mask)
        
        # Classify: [B, D] → [B, num_classes]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        
        return logits
    
    def _temporal_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Pool temporal dimension to get fixed-size representation."""
        B, T, D = embeddings.shape
        
        if self.pooling == "mean":
            return self._masked_mean_pool(embeddings, lengths, mask)
        elif self.pooling == "max":
            return self._masked_max_pool(embeddings, mask)
        elif self.pooling == "last":
            return self._last_valid_pool(embeddings, lengths)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
    
    def _masked_mean_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor],
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Mean pooling that respects sequence lengths."""
        B, T, D = embeddings.shape
        
        if lengths is not None:
            # Create mask from lengths: [B, T]
            mask = torch.arange(T, device=embeddings.device).expand(B, T) < lengths.unsqueeze(1)
        
        if mask is not None:
            # Expand mask for broadcasting: [B, T, 1]
            mask_expanded = mask.unsqueeze(-1).float()
            
            # Sum only valid positions
            summed = (embeddings * mask_expanded).sum(dim=1)  # [B, D]
            
            # Divide by actual lengths
            if lengths is not None:
                counts = lengths.unsqueeze(-1).float()  # [B, 1]
            else:
                counts = mask_expanded.sum(dim=1)  # [B, 1]
            
            # Avoid division by zero
            counts = counts.clamp(min=1.0)
            
            return summed / counts
        else:
            # No mask: simple mean
            return embeddings.mean(dim=1)
    
    def _masked_max_pool(
        self,
        embeddings: torch.Tensor,
        mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Max pooling that ignores padding."""
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            embeddings = embeddings.masked_fill(~mask_expanded, float('-inf'))
        return embeddings.max(dim=1).values
    
    def _last_valid_pool(
        self,
        embeddings: torch.Tensor,
        lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get embedding at last valid timestep."""
        B, T, D = embeddings.shape
        
        if lengths is not None:
            indices = (lengths - 1).clamp(0, T - 1)
            indices_expanded = indices.view(B, 1, 1).expand(B, 1, D)
            pooled = embeddings.gather(1, indices_expanded).squeeze(1)
            return pooled
        else:
            return embeddings[:, -1, :]
    
    def predict(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with probabilities."""
        logits = self.forward(hand, body, face, lengths)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits


# Alias for backward compatibility with loader
SignLanguageModel = SignLanguageClassifier
