# =====================================
# Data Utilities
# Multitask Sentiment & Topic Analysis
# Vietnamese Student Feedback
# =====================================

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional


class VNStudentFeedbackDataset(Dataset):
    """
    Dataset for Vietnamese student feedback with sentiment and topic labels.
    
    Args:
        texts: List of feedback texts
        sentiment_ids: List of sentiment class ids (0=negative, 1=neutral, 2=positive)
        topic_ids: List of topic class ids or list of lists for multi-label
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (default: 256)
    """
    
    def __init__(
        self,
        texts: List[str],
        sentiment_ids: List[int],
        topic_ids: List,
        tokenizer,
        max_length: int = 256
    ):
        self.texts = texts
        self.sentiment_ids = sentiment_ids
        self.topic_ids = topic_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        sent_id = self.sentiment_ids[idx]
        topic_id = self.topic_ids[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Convert topic to multi-hot if needed
        if isinstance(topic_id, (list, tuple)):
            topic_label = torch.zeros(4, dtype=torch.float32)
            for tid in topic_id:
                topic_label[tid] = 1.0
        else:
            topic_label = torch.zeros(4, dtype=torch.float32)
            topic_label[topic_id] = 1.0

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'sentiment_label': torch.tensor(sent_id, dtype=torch.long),
            'topic_label': topic_label
        }


def load_processed_data(
    data_path: str,
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Load processed CSV data.
    
    Args:
        data_path: Path to CSV file (e.g., 'datasets/processed/train.csv')
        encoding: File encoding (default: 'utf-8')
    
    Returns:
        DataFrame with columns: sentence, sentiment_id, sentiment, topic_id, topic
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, encoding=encoding)
    
    # Validate required columns
    required_cols = ['sentence', 'sentiment_id', 'topic_id']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def create_label_mappings(df: pd.DataFrame) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Create label mappings from dataframe.
    
    Args:
        df: DataFrame with sentiment and topic columns
    
    Returns:
        sent_map: {0: 'negative', 1: 'neutral', 2: 'positive'}
        topic_map: {0: 'lecturer', 1: 'training_program', 2: 'facility', 3: 'others'}
    """
    # Sentiment mapping
    if 'sentiment' in df.columns:
        sent_unique = df[['sentiment_id', 'sentiment']].drop_duplicates()
        sent_map = dict(zip(sent_unique['sentiment_id'], sent_unique['sentiment']))
    else:
        # Default mapping
        sent_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    # Topic mapping
    if 'topic' in df.columns:
        topic_unique = df[['topic_id', 'topic']].drop_duplicates()
        topic_map = dict(zip(topic_unique['topic_id'], topic_unique['topic']))
    else:
        # Default mapping
        topic_map = {0: 'lecturer', 1: 'training_program', 2: 'facility', 3: 'others'}
    
    return sent_map, topic_map


def prepare_data_for_training(
    train_path: str,
    val_path: str,
    test_path: str,
    tokenizer,
    batch_size: int = 16,
    max_length: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[int, str], Dict[int, str]]:
    """
    Prepare train, validation, and test dataloaders.
    
    Args:
        train_path: Path to train.csv
        val_path: Path to validation.csv
        test_path: Path to test.csv
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
    
    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        sent_map: Sentiment label mapping
        topic_map: Topic label mapping
    """
    # Load data
    train_df = load_processed_data(train_path)
    val_df = load_processed_data(val_path)
    test_df = load_processed_data(test_path)
    
    # Create label mappings from train data
    sent_map, topic_map = create_label_mappings(train_df)
    
    # Create datasets
    train_dataset = VNStudentFeedbackDataset(
        texts=train_df['sentence'].tolist(),
        sentiment_ids=train_df['sentiment_id'].tolist(),
        topic_ids=train_df['topic_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    val_dataset = VNStudentFeedbackDataset(
        texts=val_df['sentence'].tolist(),
        sentiment_ids=val_df['sentiment_id'].tolist(),
        topic_ids=val_df['topic_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    test_dataset = VNStudentFeedbackDataset(
        texts=test_df['sentence'].tolist(),
        sentiment_ids=test_df['sentiment_id'].tolist(),
        topic_ids=test_df['topic_id'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, sent_map, topic_map


def load_test_data_for_inference(
    test_path: str
) -> Tuple[List[str], List[int], List[int]]:
    """
    Load test data for inference.
    
    Args:
        test_path: Path to test.csv
    
    Returns:
        texts: List of sentences
        sentiment_ids: List of true sentiment labels
        topic_ids: List of true topic labels
    """
    df = load_processed_data(test_path)
    return (
        df['sentence'].tolist(),
        df['sentiment_id'].tolist(),
        df['topic_id'].tolist()
    )


def get_data_statistics(data_path: str) -> Dict:
    """
    Get statistics about the dataset.
    
    Args:
        data_path: Path to CSV file
    
    Returns:
        Dictionary with dataset statistics
    """
    df = load_processed_data(data_path)
    
    stats = {
        'total_samples': len(df),
        'sentiment_distribution': df['sentiment_id'].value_counts().to_dict(),
        'topic_distribution': df['topic_id'].value_counts().to_dict(),
        'avg_length': df['sentence'].str.len().mean(),
        'max_length': df['sentence'].str.len().max(),
        'min_length': df['sentence'].str.len().min()
    }
    
    if 'sentiment' in df.columns:
        stats['sentiment_labels'] = df['sentiment'].value_counts().to_dict()
    
    if 'topic' in df.columns:
        stats['topic_labels'] = df['topic'].value_counts().to_dict()
    
    return stats
 
