#TODO: Remove irrelevan libraries
#TODO: Combine all possible imports to 1 line

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
    print('[*] Prepping Data...')

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
    print('[+] Test set formed!')

    train_data, validation_data = train_data.split(
        split_ratio=0.8,
        stratified=True,
        strata_field='label',
        random_state= random.seed(SEED)
    )
    print('[+] Train and Validation sets formed!')
    # print(vars(train_data.examples[0]))

    print('[+] Data prepped successfully!')
    print('[*] Retrieving pretrained word embeddings...')
    TEXT.build_vocab(train_data, min_freq=1)
    embeddings = embed_prep(TEXT)
    print('[+] Embeddings retrieved successfully!')

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

    return train_data, validation_data, test_data, LABEL, TEXT, embeddings


def embed_prep(TEXT):
    # Loading fasttext model
    print('[*] Loading fasttext model...')
    fatter_fasttext_bin = load_facebook_model('crawl-300d-2M-subword/crawl-300d-2M-subword.bin')
    embedding_dim = fatter_fasttext_bin.wv.vector_size
    print('[+] Model loaded!')

    # Build embedding matrix aligned to TEXT.vocab
    num_tokens = len(TEXT.vocab)
    emb_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)

    # torchtext 0.4.0: TEXT.vocab.itos is index->token, stoi is token->index
    pad_tok = TEXT.pad_token
    unk_tok = TEXT.unk_token

    # Getting index of <unk> in vocab
    unk_index = TEXT.vocab.stoi[TEXT.unk_token]
    known_vecs = []

    print('[*] Forming embedding matrix...')
    for idx, token in enumerate(TEXT.vocab.itos):
        # Skip specials here; we will set them explicitly below
        if token in {pad_tok, unk_tok}:
            continue

        vec = fatter_fasttext_bin.wv[token]
        emb_matrix[idx] = vec
        known_vecs.append(vec)

    if len(known_vecs) > 0:
        unk_mean = torch.tensor(np.mean(known_vecs, axis=0), dtype=torch.float32)
    else:
        unk_mean = torch.empty(embedding_dim).uniform_(-0.05, 0.05)
    with torch.no_grad():
        emb_matrix[unk_index] = unk_mean

    # Create Embedding layer initialized with FastText
    fatter_embedding = torch.nn.Embedding(num_tokens, embedding_dim, padding_idx=TEXT.vocab.stoi[TEXT.pad_token])
    fatter_embedding.weight.data.copy_(torch.from_numpy(emb_matrix))
    print('[+] Embedding matrix formed!')

    return fatter_embedding


#TBC: Include function to download fasttext model
#TBC: Update function to unzip file automatically

