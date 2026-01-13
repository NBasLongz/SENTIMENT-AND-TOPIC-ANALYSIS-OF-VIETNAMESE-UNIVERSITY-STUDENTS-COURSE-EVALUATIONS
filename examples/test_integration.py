#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Script for Model Loading and Inference
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model_utils import load_model
from src.data_utils import load_processed_data, get_data_statistics
from src.eval_utils import evaluate_model, print_metrics_summary
import torch


def test_model_loading():
    """Test loading PhoBERT and mBERT models"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)
    
    models_to_test = [
        "outputs/models/phobert_multitask/final_model",
        "outputs/models/mbert_multitask/final_model"
    ]
    
    for model_dir in models_to_test:
        print(f"\nTesting: {model_dir}")
        try:
            if not os.path.exists(model_dir):
                print(f"  ⚠️  Directory not found, skipping...")
                continue
                
            model, tokenizer, sent_map, topic_map, config = load_model(
                model_dir, 
                device="cpu"
            )
            print(f"  ✅ Model loaded successfully!")
            print(f"     Sentiment classes: {len(sent_map)}")
            print(f"     Topic classes: {len(topic_map)}")
            print(f"     Backbone: {config['backbone']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_data_loading():
    """Test loading processed datasets"""
    print("\n" + "="*60)
    print("TEST 2: Data Loading")
    print("="*60)
    
    data_files = [
        "datasets/processed/train.csv",
        "datasets/processed/validation.csv",
        "datasets/processed/test.csv"
    ]
    
    for data_file in data_files:
        print(f"\nTesting: {data_file}")
        try:
            if not os.path.exists(data_file):
                print(f"  ⚠️  File not found, skipping...")
                continue
                
            df = load_processed_data(data_file)
            stats = get_data_statistics(data_file)
            
            print(f"  ✅ Data loaded successfully!")
            print(f"     Total samples: {stats['total_samples']}")
            print(f"     Sentiment distribution: {stats['sentiment_distribution']}")
            print(f"     Topic distribution: {stats['topic_distribution']}")
            print(f"     Avg sentence length: {stats['avg_length']:.1f} chars")
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_inference():
    """Test inference on sample text"""
    print("\n" + "="*60)
    print("TEST 3: Inference")
    print("="*60)
    
    model_dir = "outputs/models/phobert_multitask/final_model"
    
    if not os.path.exists(model_dir):
        print(f"⚠️  Model directory not found: {model_dir}")
        return
    
    try:
        # Load model
        model, tokenizer, sent_map, topic_map, config = load_model(
            model_dir, 
            device="cpu"
        )
        print("✅ Model loaded")
        
        # Test text
        test_text = "Giảng viên nhiệt tình, phòng học tốt"
        print(f"\nTest text: '{test_text}'")
        
        # Tokenize
        inputs = tokenizer(
            test_text,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )
        
        # Inference
        with torch.no_grad():
            sent_logits, topic_logits = model(**inputs)
        
        sent_pred = int(sent_logits.argmax(-1).item())
        topic_probs = torch.sigmoid(topic_logits).cpu().numpy()[0]
        
        print(f"\n✅ Inference successful!")
        print(f"   Sentiment: {sent_map[sent_pred]}")
        print(f"   Topic probabilities: {topic_probs}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "RUNNING ALL TESTS")
    print("="*70)
    
    test_model_loading()
    test_data_loading()
    test_inference()
    
    print("\n" + "="*70)
    print(" "*20 + "TESTS COMPLETED")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
