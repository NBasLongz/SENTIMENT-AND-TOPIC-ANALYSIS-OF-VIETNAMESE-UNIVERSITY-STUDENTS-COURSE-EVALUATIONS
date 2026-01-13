# =====================================
# Model Utilities
# Multitask Sentiment & Topic Analysis
# Vietnamese Student Feedback
# =====================================

import os
import json
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, Optional


class MultiTaskModel(nn.Module):
    """
    Multi-task BERT model for simultaneous sentiment classification 
    and multi-label topic classification.
    
    Args:
        encoder_name: Pretrained model name (e.g., 'vinai/phobert-base', 'bert-base-multilingual-cased')
        n_sent: Number of sentiment classes (default: 3)
        n_topic: Number of topic classes (default: 4)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, encoder_name: str, n_sent: int = 3, n_topic: int = 4, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.sent_head = nn.Linear(hidden, n_sent)
        self.topic_head = nn.Linear(hidden, n_topic)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, return_embedding=False, **kwargs):
        """
        Forward pass for multi-task model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (optional, unused but accepted for compatibility)
            return_embedding: Whether to return the CLS embedding vector
            **kwargs: Other arguments (ignored)
        
        Returns:
            sent_logits: Sentiment logits (batch_size, n_sent)
            topic_logits: Topic logits (batch_size, n_topic)
            cls_embedding: CLS token embedding (batch_size, hidden_size) - only if return_embedding=True
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.drop(out.last_hidden_state[:, 0, :])
        sent_logits = self.sent_head(cls)
        topic_logits = self.topic_head(cls)
        
        if return_embedding:
            return sent_logits, topic_logits, cls
            
        return sent_logits, topic_logits


def load_model(
    model_dir: str,
    device: str = "cpu"
) -> Tuple[MultiTaskModel, AutoTokenizer, Dict[int, str], Dict[int, str], Dict]:
    """
    Load trained multi-task model from directory.
    
    Args:
        model_dir: Path to model directory (e.g., 'outputs/models/phobert_multitask/final_model')
        device: Device to load model on ('cpu' or 'cuda')
    
    Returns:
        model: Loaded MultiTaskModel
        tokenizer: Tokenizer for the model
        sent_map: Sentiment label mapping {0: 'negative', 1: 'neutral', 2: 'positive'}
        topic_map: Topic label mapping {0: 'lecturer', 1: 'training_program', ...}
        config: Model configuration dictionary
    """
    # Load label maps
    label_maps_path = os.path.join(model_dir, "label_maps.json")
    if not os.path.exists(label_maps_path):
        raise FileNotFoundError(f"label_maps.json not found in {model_dir}")
    
    with open(label_maps_path, "r", encoding="utf-8") as f:
        maps = json.load(f)

    sent_map = {int(k): v for k, v in maps["sentiment"].items()}
    topic_map = {int(k): v for k, v in maps["topic"].items()}

    # Determine backbone model
    if "phobert" in model_dir.lower():
        backbone = "vinai/phobert-base"
    elif "mbert" in model_dir.lower():
        backbone = "bert-base-multilingual-cased"
    else:
        # Try to infer from tokenizer config
        tokenizer_config_path = os.path.join(model_dir, "tokenizer_config.json")
        if os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)
                backbone = tok_cfg.get("model_max_length", "bert-base-multilingual-cased")
        else:
            backbone = "bert-base-multilingual-cased"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)

    # Initialize model
    model = MultiTaskModel(
        encoder_name=backbone,
        n_sent=len(sent_map),
        n_topic=len(topic_map)
    )

    # Load model weights
    model_path = os.path.join(model_dir, "pytorch_model.bin")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"pytorch_model.bin not found in {model_dir}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Create config dict
    config = {
        "backbone": backbone,
        "n_sent": len(sent_map),
        "n_topic": len(topic_map),
        "model_dir": model_dir
    }

    return model, tokenizer, sent_map, topic_map, config


def save_model(
    model: MultiTaskModel,
    tokenizer: AutoTokenizer,
    save_dir: str,
    sent_map: Dict[int, str],
    topic_map: Dict[int, str]
):
    """
    Save multi-task model and associated files.
    
    Args:
        model: MultiTaskModel to save
        tokenizer: Tokenizer to save
        save_dir: Directory to save model files
        sent_map: Sentiment label mapping
        topic_map: Topic label mapping
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model weights
    torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_dir)
    
    # Save label maps
    label_maps = {
        "sentiment": {str(k): v for k, v in sent_map.items()},
        "topic": {str(k): v for k, v in topic_map.items()}
    }
    with open(os.path.join(save_dir, "label_maps.json"), "w", encoding="utf-8") as f:
        json.dump(label_maps, f, ensure_ascii=False, indent=2)
    
    print(f"Model saved to {save_dir}")
 
