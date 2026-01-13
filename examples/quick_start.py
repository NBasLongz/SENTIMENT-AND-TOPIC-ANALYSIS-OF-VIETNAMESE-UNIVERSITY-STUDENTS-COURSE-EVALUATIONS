#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick Start Example
Vietnamese Student Feedback Analysis
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_model
import torch
import numpy as np


def main():
    """Quick start example for using the trained models"""
    
    print("="*60)
    print("Vietnamese Student Feedback Analysis - Quick Start")
    print("="*60)
    
    # Configuration
    MODEL_DIR = "outputs/models/phobert_multitask/final_model"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOPIC_THRESHOLD = 0.35
    
    print(f"\nDevice: {DEVICE}")
    print(f"Loading model from: {MODEL_DIR}")
    
    # Load model
    try:
        model, tokenizer, sent_map, topic_map, config = load_model(MODEL_DIR, device=DEVICE)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Example texts
    examples = [
        "Giảng viên nhiệt tình, giảng dạy dễ hiểu",
        "Phòng học nóng quá, máy lạnh hư",
        "Chương trình học quá nặng, deadline nhiều",
        "Thầy giảng hay nhưng phòng học xa quá"
    ]
    
    print("\n" + "="*60)
    print("Running inference on example texts:")
    print("="*60)
    
    for i, text in enumerate(examples, 1):
        print(f"\n[Example {i}]")
        print(f"Text: {text}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            sent_logits, topic_logits = model(**inputs)
        
        # Sentiment prediction
        sent_pred = int(sent_logits.argmax(-1).item())
        sent_label = sent_map[sent_pred]
        sent_probs = torch.softmax(sent_logits, dim=-1).cpu().numpy()[0]
        
        # Topic prediction
        topic_probs = torch.sigmoid(topic_logits).cpu().numpy()[0]
        topic_preds = np.where(topic_probs >= TOPIC_THRESHOLD)[0]
        
        # If no topic above threshold, take the highest
        if len(topic_preds) == 0:
            topic_preds = [topic_probs.argmax()]
        
        topic_labels = [topic_map[idx] for idx in topic_preds]
        
        # Display results
        print(f"→ Sentiment: {sent_label} (confidence: {sent_probs[sent_pred]:.3f})")
        print(f"→ Topics: {', '.join(topic_labels)}")
        print(f"  Topic probabilities:")
        for idx, name in topic_map.items():
            print(f"    {name:20s}: {topic_probs[idx]:.3f}")
    
    print("\n" + "="*60)
    print("Demo completed!")
    print("="*60)


if __name__ == "__main__":
    main()
