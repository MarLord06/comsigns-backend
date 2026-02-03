"""
Prediction utilities for inference.

Runs model inference and returns structured predictions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .model import SignLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class TopKPrediction:
    """A single prediction in top-k results.
    
    Attributes:
        class_id: Predicted class ID
        class_name: Class name (if available)
        score: Confidence score (probability)
        rank: Rank in top-k (1-indexed)
    """
    class_id: int
    class_name: str
    score: float
    rank: int


@dataclass
class PredictionResult:
    """Result of a prediction.
    
    Attributes:
        top1_class_id: Top-1 predicted class ID
        top1_class_name: Top-1 class name
        top1_score: Top-1 confidence score
        topk: List of top-k predictions
        logits: Raw logits tensor (optional)
    """
    top1_class_id: int
    top1_class_name: str
    top1_score: float
    topk: List[TopKPrediction]
    logits: Optional[torch.Tensor] = None


class Predictor:
    """Predictor for running inference.
    
    Wraps a model and provides convenient prediction methods.
    
    Example:
        >>> predictor = Predictor(model, class_names, device="cpu")
        >>> result = predictor.predict_from_features({
        ...     "hand": hand_tensor,
        ...     "body": body_tensor,
        ...     "face": face_tensor
        ... })
        >>> print(result.top1_class_name, result.top1_score)
    
    Attributes:
        model: The loaded model
        class_names: List of class names
        other_class_id: ID of OTHER class (if any)
        device: Inference device
        topk: Number of top predictions to return
    """
    
    def __init__(
        self,
        model: SignLanguageModel,
        class_names: List[str],
        other_class_id: Optional[int] = None,
        device: str = "cpu",
        topk: int = 5
    ):
        """Initialize predictor.
        
        Args:
            model: Trained SignLanguageModel
            class_names: List of class names by index
            other_class_id: ID of OTHER class (if tail_to_other)
            device: Device for inference
            topk: Default number of top predictions
        """
        self.model = model
        # Ensure class_names is a list (handle dict format)
        if isinstance(class_names, dict):
            max_idx = max(int(k) for k in class_names.keys())
            self.class_names = [
                class_names.get(i, class_names.get(str(i), f"class_{i}"))
                for i in range(max_idx + 1)
            ]
        else:
            self.class_names = list(class_names)
        self.other_class_id = other_class_id
        self.device = device
        self._topk = topk
        
        # Ensure model is in eval mode
        self.model.eval()
    
    @property
    def topk(self) -> int:
        """Get top-k value."""
        return self._topk
    
    @topk.setter
    def topk(self, value: int) -> None:
        """Set top-k value."""
        if value < 1:
            raise ValueError("topk must be >= 1")
        self._topk = min(value, len(self.class_names))
    
    def _prepare_input(
        self,
        features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Prepare input features for model.
        
        Args:
            features: Dict with hand, body, face tensors
        
        Returns:
            Prepared features on correct device with batch dim
        """
        prepared = {}
        
        for key in ['hand', 'body', 'face']:
            tensor = features[key]
            
            # Ensure float32
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            
            # Add batch dimension if needed
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            
            # Move to device
            tensor = tensor.to(self.device)
            prepared[key] = tensor
        
        # Handle lengths if present
        if 'lengths' in features:
            lengths = features['lengths']
            if not isinstance(lengths, torch.Tensor):
                lengths = torch.tensor(lengths, dtype=torch.long)
            if lengths.dim() == 0:
                lengths = lengths.unsqueeze(0)
            prepared['lengths'] = lengths.to(self.device)
        else:
            # Use full sequence length
            seq_len = prepared['hand'].shape[1]
            prepared['lengths'] = torch.tensor([seq_len], dtype=torch.long)
        
        return prepared
    
    @torch.no_grad()
    def predict_from_features(
        self,
        features: Dict[str, torch.Tensor],
        return_logits: bool = False
    ) -> PredictionResult:
        """Run prediction on prepared features.
        
        Args:
            features: Dict with hand, body, face tensors
            return_logits: Whether to include logits in result
        
        Returns:
            PredictionResult with top-k predictions
        """
        # Prepare input
        prepared = self._prepare_input(features)
        
        # Forward pass
        logits = self.model(
            hand=prepared['hand'],
            body=prepared['body'],
            face=prepared['face'],
            lengths=prepared.get('lengths')
        )
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get top-k
        k = min(self._topk, probs.shape[-1])
        topk_probs, topk_indices = torch.topk(probs[0], k)
        
        # Build predictions
        topk_preds = []
        for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices), start=1):
            class_id = idx.item()
            topk_preds.append(TopKPrediction(
                class_id=class_id,
                class_name=self.class_names[class_id],
                score=prob.item(),
                rank=rank
            ))
        
        # Top-1
        top1 = topk_preds[0]
        
        return PredictionResult(
            top1_class_id=top1.class_id,
            top1_class_name=top1.class_name,
            top1_score=top1.score,
            topk=topk_preds,
            logits=logits if return_logits else None
        )
    
    def predict_batch(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> List[PredictionResult]:
        """Run prediction on a batch.
        
        Args:
            batch: Batched tensors with hand, body, face, lengths
        
        Returns:
            List of PredictionResult, one per sample
        """
        # Prepare batch
        prepared = {}
        for key in ['hand', 'body', 'face']:
            tensor = batch[key]
            if tensor.dtype != torch.float32:
                tensor = tensor.float()
            prepared[key] = tensor.to(self.device)
        
        if 'lengths' in batch:
            prepared['lengths'] = batch['lengths'].to(self.device)
        else:
            prepared['lengths'] = None
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(
                hand=prepared['hand'],
                body=prepared['body'],
                face=prepared['face'],
                lengths=prepared['lengths']
            )
            probs = torch.softmax(logits, dim=-1)
        
        # Build results for each sample
        results = []
        batch_size = probs.shape[0]
        k = min(self._topk, probs.shape[-1])
        
        for i in range(batch_size):
            topk_probs, topk_indices = torch.topk(probs[i], k)
            
            topk_preds = []
            for rank, (prob, idx) in enumerate(zip(topk_probs, topk_indices), start=1):
                class_id = idx.item()
                topk_preds.append(TopKPrediction(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    score=prob.item(),
                    rank=rank
                ))
            
            top1 = topk_preds[0]
            results.append(PredictionResult(
                top1_class_id=top1.class_id,
                top1_class_name=top1.class_name,
                top1_score=top1.score,
                topk=topk_preds
            ))
        
        return results
