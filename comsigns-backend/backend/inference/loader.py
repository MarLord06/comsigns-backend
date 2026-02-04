"""
Model loading utilities for inference.

Loads trained checkpoints and class mappings.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

import torch

from .model import SignLanguageModel

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model.
    
    Attributes:
        num_classes: Number of output classes
        checkpoint_path: Path to checkpoint file
        epoch: Training epoch of checkpoint
        tail_to_other: Whether tail classes are mapped to OTHER
        class_names: List of class names by index
        other_class_id: ID of the OTHER class (if tail_to_other)
    """
    num_classes: int
    checkpoint_path: Path
    epoch: int
    tail_to_other: bool
    class_names: List[str]
    other_class_id: Optional[int] = None


class InferenceLoader:
    """Loader for inference models and class mappings.
    
    Handles:
    - Loading PyTorch checkpoints
    - Parsing class_mapping.json
    - Building class name lists
    
    Example:
        >>> loader = InferenceLoader(
        ...     checkpoint_path="models/run/checkpoints/best.pt",
        ...     class_mapping_path="models/run/class_mapping.json",
        ...     device="cpu"
        ... )
        >>> model = loader.load_model()
        >>> class_names = loader.get_class_names()
    
    Attributes:
        checkpoint_path: Path to .pt checkpoint
        class_mapping_path: Path to class_mapping.json
        device: Target device
        model_info: Loaded model info (after load_model)
    """
    
    def __init__(
        self,
        checkpoint_path: Path,
        class_mapping_path: Path,
        device: str = "cpu"
    ):
        """Initialize the loader.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            class_mapping_path: Path to class_mapping.json
            device: Device to load model to ("cpu" or "cuda")
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.class_mapping_path = Path(class_mapping_path)
        self.device = device
        
        self._model: Optional[SignLanguageModel] = None
        self._model_info: Optional[ModelInfo] = None
        self._class_mapping: Optional[Dict[str, Any]] = None
        self._class_names: Optional[List[str]] = None
    
    @property
    def model_info(self) -> ModelInfo:
        """Get model info (loads if needed)."""
        if self._model_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model_info
    
    def _load_class_mapping(self) -> Dict[str, Any]:
        """Load and parse class_mapping.json."""
        if self._class_mapping is not None:
            return self._class_mapping
        
        with open(self.class_mapping_path, 'r') as f:
            self._class_mapping = json.load(f)
        
        return self._class_mapping
    
    def _build_class_names(self, num_classes: int) -> List[str]:
        """Build list of class names from mapping.
        
        Args:
            num_classes: Number of classes in the model
        
        Returns:
            List of class names indexed by new_class_id
        """
        if self._class_names is not None:
            return self._class_names
        
        mapping = self._load_class_mapping()
        
        # Check if new_class_names exists in mapping
        if 'new_class_names' in mapping:
            names = mapping['new_class_names']
            # Handle dict format (e.g., {"0": "name0", "1": "name1", ...})
            if isinstance(names, dict):
                self._class_names = [
                    names.get(i, names.get(str(i), f"class_{i}"))
                    for i in range(num_classes)
                ]
            else:
                self._class_names = list(names)
            return self._class_names
        
        # Otherwise build from old_to_new mapping
        # We'll use the new_class_id as the name (numeric)
        # The semantic resolver will handle gloss resolution
        self._class_names = [str(i) for i in range(num_classes)]
        
        # Check if we have OTHER class
        stats = mapping.get('statistics', {})
        other_id = stats.get('other_class_id')
        if other_id is not None:
            config = mapping.get('config', {})
            other_name = config.get('other_class_name', 'OTHER')
            self._class_names[other_id] = other_name
        
        return self._class_names
    
    def load_model(self) -> SignLanguageModel:
        """Load the model from checkpoint.
        
        Returns:
            Loaded SignLanguageModel in eval mode
        
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            RuntimeError: If checkpoint format is invalid
        """
        if self._model is not None:
            return self._model
        
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        logger.info(f"Loading checkpoint: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )
        
        # Extract metadata
        num_classes = checkpoint.get('num_classes', 142)
        epoch = checkpoint.get('epoch', 0)
        tail_to_other = checkpoint.get('tail_to_other', False)
        
        # Get model state dict
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            raise RuntimeError("Checkpoint missing model state dict")
        
        # Infer input dimensions from state dict
        hand_dim = state_dict['encoder.hand_branch.input_proj.weight'].shape[1]
        body_dim = state_dict['encoder.body_branch.input_proj.weight'].shape[1]
        face_dim = state_dict['encoder.face_branch.input_proj.0.weight'].shape[1]
        hidden_dim = state_dict['encoder.hand_branch.input_proj.weight'].shape[0]
        
        logger.info(
            f"Model config: hand_dim={hand_dim}, body_dim={body_dim}, "
            f"face_dim={face_dim}, hidden_dim={hidden_dim}, num_classes={num_classes}"
        )
        
        # Create model
        self._model = SignLanguageModel(
            hand_dim=hand_dim,
            body_dim=body_dim,
            face_dim=face_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes
        )
        
        # Load state dict
        self._model.load_state_dict(state_dict)
        self._model.to(self.device)
        self._model.eval()
        
        # Build class names
        class_names = self._build_class_names(num_classes)
        
        # Get OTHER class ID
        mapping = self._load_class_mapping()
        other_class_id = mapping.get('statistics', {}).get('other_class_id')
        
        # Store model info
        self._model_info = ModelInfo(
            num_classes=num_classes,
            checkpoint_path=self.checkpoint_path,
            epoch=epoch,
            tail_to_other=tail_to_other,
            class_names=class_names,
            other_class_id=other_class_id
        )
        
        logger.info(
            f"Model loaded: {num_classes} classes, epoch={epoch}, "
            f"tail_to_other={tail_to_other}, device={self.device}"
        )
        
        return self._model
    
    def get_class_names(self) -> List[str]:
        """Get list of class names.
        
        Returns:
            List of class names indexed by class ID
        
        Raises:
            RuntimeError: If model not loaded
        """
        if self._model_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model_info.class_names
    
    def get_other_class_id(self) -> Optional[int]:
        """Get the OTHER class ID if tail_to_other is enabled.
        
        Returns:
            OTHER class ID or None
        """
        if self._model_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model_info.other_class_id
    
    def unload(self) -> None:
        """Unload model from memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._model_info = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Model unloaded")
