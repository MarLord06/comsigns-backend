"""
Sign language recognition model architecture.

Multi-branch LSTM model with hand, body, and face modalities.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


class ModalityBranch(nn.Module):
    """Branch for processing a single modality (hand, body, or face).
    
    Architecture:
        Input projection → LSTM → LayerNorm
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        use_mlp_proj: Use MLP for input projection (for high-dim inputs like face)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_mlp_proj: bool = False
    ):
        super().__init__()
        
        # Input projection
        if use_mlp_proj:
            # MLP projection for high-dimensional inputs (face)
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        else:
            # Simple linear projection
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, input_dim]
            lengths: Sequence lengths [batch] (optional)
        
        Returns:
            Output tensor [batch, hidden_dim]
        """
        # Project input
        x = self.input_proj(x)
        
        # Pack if lengths provided
        if lengths is not None:
            # Ensure lengths are on CPU for pack_padded_sequence
            lengths_cpu = lengths.cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)
        
        # Unpack if needed
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # Get last hidden state
        # h_n: [num_layers, batch, hidden_dim]
        last_hidden = h_n[-1]  # [batch, hidden_dim]
        
        # Layer normalization
        output = self.layer_norm(last_hidden)
        
        return output


class SignLanguageModel(nn.Module):
    """Multi-branch sign language recognition model.
    
    Architecture:
        - Hand branch: LSTM on hand keypoints
        - Body branch: LSTM on body/pose keypoints  
        - Face branch: LSTM on face keypoints
        - Fusion: Concatenate + MLP → classifier
    
    Args:
        hand_dim: Hand input dimension (default: 168 = 2 hands * 21 landmarks * 4 coords)
        body_dim: Body input dimension (default: 132 = 33 landmarks * 4 coords)
        face_dim: Face input dimension (default: 1872 = 468 landmarks * 4 coords)
        hidden_dim: Hidden dimension for all branches
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        hand_dim: int = 168,
        body_dim: int = 132,
        face_dim: int = 1872,
        hidden_dim: int = 256,
        num_classes: int = 142,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Encoder branches
        self.encoder = nn.ModuleDict({
            'hand_branch': ModalityBranch(
                input_dim=hand_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                use_mlp_proj=False
            ),
            'body_branch': ModalityBranch(
                input_dim=body_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                use_mlp_proj=False
            ),
            'face_branch': ModalityBranch(
                input_dim=face_dim,
                hidden_dim=hidden_dim,
                num_layers=2,
                dropout=dropout,
                use_mlp_proj=True  # MLP for high-dim face input
            )
        })
        
        # Fusion layer
        fusion_input_dim = hidden_dim * 3  # 3 branches concatenated
        self.encoder['fusion'] = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim * 2)
        )
        
        # Classifier
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            hand: Hand features [batch, seq_len, hand_dim]
            body: Body features [batch, seq_len, body_dim]
            face: Face features [batch, seq_len, face_dim]
            lengths: Sequence lengths [batch] (optional)
        
        Returns:
            Logits [batch, num_classes]
        """
        # Process each branch
        hand_out = self.encoder['hand_branch'](hand, lengths)
        body_out = self.encoder['body_branch'](body, lengths)
        face_out = self.encoder['face_branch'](face, lengths)
        
        # Concatenate
        fused = torch.cat([hand_out, body_out, face_out], dim=-1)
        
        # Fusion MLP
        fused = self.encoder['fusion'](fused)
        
        # Classify
        logits = self.classifier(fused)
        
        return logits
    
    def predict(
        self,
        hand: torch.Tensor,
        body: torch.Tensor,
        face: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with probabilities.
        
        Args:
            hand: Hand features
            body: Body features
            face: Face features
            lengths: Sequence lengths
        
        Returns:
            Tuple of (probabilities, logits)
        """
        logits = self.forward(hand, body, face, lengths)
        probs = torch.softmax(logits, dim=-1)
        return probs, logits
