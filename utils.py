import os

os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch

# Method 2: Patch torch._dynamo.disable decorator after import
try:
    import torch._dynamo
    # Patch the disable function to ignore the 'wrapping' parameter
    if hasattr(torch._dynamo, 'disable'):
        def patched_disable(fn=None, *args, **kwargs):
            # Remove problematic 'wrapping' parameter if present
            if 'wrapping' in kwargs:
                kwargs.pop('wrapping')
            if fn is None:
                # Decorator usage: @disable
                return lambda f: f
            # Function usage: disable(fn) or disable(fn, **kwargs)
            # Simply return the function unwrapped to avoid recursion
            # The original disable was causing issues, so we bypass it entirely
            return fn
        torch._dynamo.disable = patched_disable
except Exception as e:
    print(f"Warning: Could not patch torch._dynamo: {e}")
    pass  # If patching fails, continue anyway

import random, string

from torchtext import data , datasets
from collections import defaultdict, Counter
import numpy as np

os.environ['GENSIM_DATA_DIR'] = os.path.join(os.getcwd(), 'gensim-data')

import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.models.fasttext import load_facebook_model

from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import time, copy


def data_prep(SEED = 42):
    ### Part 0: Dataset Preparation

    # For tokenization
    TEXT = data.Field ( tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', include_lengths = True )

    # For multi - class classification labels
    LABEL = data.LabelField ()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load the TREC dataset
    # Train / Validation / Test split
    train_data, test_data = datasets.TREC.splits( TEXT, LABEL, fine_grained = False )

    train_data, validation_data = train_data.split(
        split_ratio=0.8,
        stratified=True,
        strata_field='label',
        random_state= random.seed(SEED)
    )
    print(vars(train_data.examples[0]))


    # Count how many samples per label in the train set
    label_counts = Counter([ex.label for ex in train_data.examples])
    total_examples = len(train_data)

    print("\nLabel distribution in training set:")
    for label, count in sorted(label_counts.items()):
        percentage = (count / total_examples) * 100
        print(f"- {label}: {count} samples ({percentage:.2f}%)")

    # Optional sanity check: total percentages should sum ≈ 100%
    total_percentage = sum((count / total_examples) * 100 for count in label_counts.values())
    print(f"Total samples: {total_examples}, Sum of percentages: {total_percentage:.2f}%")

    return train_data, validation_data, test_data, LABEL, TEXT