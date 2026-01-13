# =====================================
# Evaluation Utilities
# Multitask Sentiment & Topic Analysis
# Vietnamese Student Feedback
# =====================================

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    hamming_loss,
    jaccard_score
)
from typing import Dict, List, Tuple, Optional
import json


def predict_batch(
    model,
    batch,
    device: str,
    topic_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions on a batch.
    
    Args:
        model: MultiTaskModel
        batch: Batch from DataLoader
        device: Device to run inference on
        topic_threshold: Threshold for multi-label topic classification
    
    Returns:
        sent_preds: Predicted sentiment labels
        topic_preds: Predicted topic labels (multi-hot)
        sent_probs: Sentiment probabilities
        topic_probs: Topic probabilities (sigmoid)
    """
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    with torch.no_grad():
        sent_logits, topic_logits = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Sentiment predictions (argmax)
    sent_probs = torch.softmax(sent_logits, dim=-1).cpu().numpy()
    sent_preds = sent_logits.argmax(dim=-1).cpu().numpy()
    
    # Topic predictions (sigmoid threshold)
    topic_probs = torch.sigmoid(topic_logits).cpu().numpy()
    topic_preds = (topic_probs >= topic_threshold).astype(int)
    
    # If no topic predicted, take the one with highest probability
    for i in range(len(topic_preds)):
        if topic_preds[i].sum() == 0:
            topic_preds[i][topic_probs[i].argmax()] = 1
    
    return sent_preds, topic_preds, sent_probs, topic_probs


def evaluate_sentiment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate sentiment classification.
    
    Args:
        y_true: True sentiment labels
        y_pred: Predicted sentiment labels
        labels: Label names (optional)
    
    Returns:
        Dictionary with metrics
    """
    if labels is None:
        labels = ['negative', 'neutral', 'positive']
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    per_class_metrics = classification_report(
        y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'per_class': per_class_metrics,
        'confusion_matrix': cm.tolist()
    }


def evaluate_topic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate multi-label topic classification.
    
    Args:
        y_true: True topic labels (multi-hot encoded)
        y_pred: Predicted topic labels (multi-hot encoded)
        labels: Label names (optional)
    
    Returns:
        Dictionary with metrics
    """
    if labels is None:
        labels = ['lecturer', 'training_program', 'facility', 'others']
    
    # Hamming loss (fraction of incorrect labels)
    hamming = hamming_loss(y_true, y_pred)
    
    # Jaccard score (average IoU)
    jaccard = jaccard_score(y_true, y_pred, average='samples', zero_division=0)
    
    # Per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and micro averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    
    per_label_metrics = {}
    for i, label in enumerate(labels):
        per_label_metrics[label] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i]),
            'support': int(support[i])
        }
    
    return {
        'hamming_loss': float(hamming),
        'jaccard_score': float(jaccard),
        'macro_precision': float(macro_precision),
        'macro_recall': float(macro_recall),
        'macro_f1': float(macro_f1),
        'micro_precision': float(micro_precision),
        'micro_recall': float(micro_recall),
        'micro_f1': float(micro_f1),
        'per_label': per_label_metrics
    }


def evaluate_model(
    model,
    dataloader,
    device: str,
    topic_threshold: float = 0.5,
    sent_labels: Optional[List[str]] = None,
    topic_labels: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: MultiTaskModel
        dataloader: DataLoader
        device: Device to run inference on
        topic_threshold: Threshold for topic classification
        sent_labels: Sentiment label names
        topic_labels: Topic label names
    
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_sent_true = []
    all_sent_pred = []
    all_topic_true = []
    all_topic_pred = []
    
    for batch in dataloader:
        # Get predictions
        sent_preds, topic_preds, _, _ = predict_batch(
            model, batch, device, topic_threshold
        )
        
        # Collect true labels
        sent_true = batch['sentiment_label'].numpy()
        topic_true = batch['topic_label'].numpy()
        
        all_sent_true.append(sent_true)
        all_sent_pred.append(sent_preds)
        all_topic_true.append(topic_true)
        all_topic_pred.append(topic_preds)
    
    # Concatenate all batches
    all_sent_true = np.concatenate(all_sent_true)
    all_sent_pred = np.concatenate(all_sent_pred)
    all_topic_true = np.concatenate(all_topic_true)
    all_topic_pred = np.concatenate(all_topic_pred)
    
    # Evaluate sentiment
    sent_metrics = evaluate_sentiment(all_sent_true, all_sent_pred, sent_labels)
    
    # Evaluate topic
    topic_metrics = evaluate_topic(all_topic_true, all_topic_pred, topic_labels)
    
    return {
        'sentiment': sent_metrics,
        'topic': topic_metrics,
        'n_samples': len(all_sent_true)
    }


def find_optimal_threshold(
    model,
    dataloader,
    device: str,
    thresholds: Optional[List[float]] = None
) -> Tuple[float, Dict]:
    """
    Find optimal threshold for topic classification on validation set.
    
    Args:
        model: MultiTaskModel
        dataloader: Validation DataLoader
        device: Device to run inference on
        thresholds: List of thresholds to try (default: 0.1 to 0.9 step 0.05)
    
    Returns:
        best_threshold: Optimal threshold value
        results: Dictionary with results for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05).tolist()
    
    model.eval()
    
    # Collect all predictions and true labels
    all_topic_probs = []
    all_topic_true = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            _, topic_logits = model(input_ids=input_ids, attention_mask=attention_mask)
        
        topic_probs = torch.sigmoid(topic_logits).cpu().numpy()
        topic_true = batch['topic_label'].numpy()
        
        all_topic_probs.append(topic_probs)
        all_topic_true.append(topic_true)
    
    all_topic_probs = np.concatenate(all_topic_probs)
    all_topic_true = np.concatenate(all_topic_true)
    
    # Try each threshold
    results = {}
    best_f1 = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        topic_preds = (all_topic_probs >= threshold).astype(int)
        
        # Handle no prediction case
        for i in range(len(topic_preds)):
            if topic_preds[i].sum() == 0:
                topic_preds[i][all_topic_probs[i].argmax()] = 1
        
        metrics = evaluate_topic(all_topic_true, topic_preds)
        
        results[float(threshold)] = {
            'macro_f1': metrics['macro_f1'],
            'micro_f1': metrics['micro_f1'],
            'jaccard': metrics['jaccard_score']
        }
        
        # Track best threshold based on macro F1
        if metrics['macro_f1'] > best_f1:
            best_f1 = metrics['macro_f1']
            best_threshold = threshold
    
    return best_threshold, results


def save_metrics(metrics: Dict, output_path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Metrics saved to {output_path}")


def print_metrics_summary(metrics: Dict):
    """
    Print formatted metrics summary.
    
    Args:
        metrics: Metrics dictionary from evaluate_model
    """
    print("\n" + "="*60)
    print("SENTIMENT CLASSIFICATION METRICS")
    print("="*60)
    sent = metrics['sentiment']
    print(f"Accuracy:  {sent['accuracy']:.4f}")
    print(f"Precision: {sent['precision']:.4f}")
    print(f"Recall:    {sent['recall']:.4f}")
    print(f"F1-Score:  {sent['f1']:.4f}")
    
    print("\n" + "="*60)
    print("TOPIC CLASSIFICATION METRICS (Multi-label)")
    print("="*60)
    topic = metrics['topic']
    print(f"Jaccard Score:  {topic['jaccard_score']:.4f}")
    print(f"Hamming Loss:   {topic['hamming_loss']:.4f}")
    print(f"Macro F1:       {topic['macro_f1']:.4f}")
    print(f"Micro F1:       {topic['micro_f1']:.4f}")
    
    print("\nPer-label metrics:")
    for label, m in topic['per_label'].items():
        print(f"  {label:20s} - P: {m['precision']:.3f}, R: {m['recall']:.3f}, F1: {m['f1']:.3f}")
    
    print("\n" + "="*60)
 
