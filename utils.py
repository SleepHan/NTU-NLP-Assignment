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


# ============================================================================
# Helper function to process batches consistently
# ============================================================================
def process_batch(batch, debug=False):
    """
    Process a batch from BucketIterator, handling text transpose correctly.
    Returns: text, text_lengths, labels (all properly formatted)
    """
    text, text_lengths = batch.text
    labels = batch.label
    
    if debug:
        print(f"DEBUG BATCH - text shape: {text.shape}, text_lengths shape: {text_lengths.shape}, labels shape: {labels.shape}")
    
    # torchtext BucketIterator returns text as [seq_len, batch_size] by default
    # We need [batch_size, seq_len] for batch_first=True in the model
    expected_batch_size = labels.shape[0]
    
    if text.dim() == 2:
        if text.shape[1] == expected_batch_size and len(text_lengths) == expected_batch_size:
            # text is [seq_len, batch_size], transpose to [batch_size, seq_len]
            text = text.transpose(0, 1)
            if debug:
                print(f"DEBUG BATCH - Transposed text to [batch_size, seq_len]: {text.shape}")
        elif text.shape[0] == expected_batch_size and len(text_lengths) == expected_batch_size:
            # text is already [batch_size, seq_len]
            if debug:
                print(f"DEBUG BATCH - text already in correct format: {text.shape}")
        else:
            raise ValueError(
                f"Cannot determine text format: text.shape={text.shape}, "
                f"text_lengths.shape={text_lengths.shape}, labels.shape={labels.shape}"
            )
    
    # Verify dimensions match
    assert text.shape[0] == len(text_lengths) == labels.shape[0], \
        f"Batch size mismatch: text.shape[0]={text.shape[0]}, len(text_lengths)={len(text_lengths)}, labels.shape[0]={labels.shape[0]}"
    
    return text, text_lengths, labels


# Utility function for counting parameters
def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Training function with history tracking for plotting curves
def train_model_with_history(model, train_iterator, val_iterator, optimizer, criterion, 
                             n_epochs, device, num_classes, patience=10, model_name="model"):
    """
    Train the model with early stopping and track training history.
    Returns: model, training_history dictionary
    """
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_val_auc = 0.0
    patience_counter = 0
    best_model_state = None
    
    # History tracking
    train_losses = []
    val_losses = []
    val_accs = []
    val_f1s = []  # Add F1 tracking
    val_aucs = []  # Add AUROC tracking
    
    print(f"\n>>> Training {model_name}")
    print(f"    Parameters: {count_parameters(model):,}")
    print(f"    Max epochs: {n_epochs}, Patience: {patience}")
    
    for epoch in range(n_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in train_iterator:
            text, text_lengths, labels = process_batch(batch, debug=False)
            optimizer.zero_grad()
            predictions = model(text, text_lengths)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = torch.argmax(predictions, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        avg_train_loss = train_loss / len(train_iterator)
        train_acc = accuracy_score(train_labels, train_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_probs = []  # Add probabilities for AUROC
        
        with torch.no_grad():
            for batch in val_iterator:
                text, text_lengths, labels = process_batch(batch, debug=False)
                predictions = model(text, text_lengths)
                loss = criterion(predictions, labels)
                val_loss += loss.item()
                
                probs = torch.softmax(predictions, dim=1)  # Get probabilities
                preds = torch.argmax(predictions, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())  # Store probabilities
        
        avg_val_loss = val_loss / len(val_iterator)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')  # Calculate F1
        
        # Calculate AUC-ROC
        try:
            val_probs_array = np.array(val_probs)
            val_labels_bin = label_binarize(val_labels, classes=range(num_classes))
            val_auc = roc_auc_score(val_labels_bin, val_probs_array, average='weighted', multi_class='ovr')
        except Exception as e:
            print(f"Warning: Could not calculate AUC-ROC for {model_name} at epoch {epoch+1}: {e}")
            val_auc = 0.0
        
        # Store history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)
        val_f1s.append(val_f1)  # Store F1
        val_aucs.append(val_auc)  # Store AUROC
        
        # Early stopping and model saving (using accuracy for early stopping)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print(f'Epoch: {epoch+1:02}/{n_epochs} | Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {avg_train_loss:.4f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\tVal Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}')
        
        if patience_counter >= patience:
            print(f'\t>>> Early stopping at epoch {epoch+1}, best val acc: {best_val_acc*100:.2f}%')
            break
    
    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f'\n>>> Training completed! Best validation accuracy: {best_val_acc*100:.2f}%')
    print(f'    Best validation F1: {best_val_f1:.4f}')
    print(f'    Best validation AUC-ROC: {best_val_auc:.4f}')
    
    # Return history dictionary
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'val_f1s': val_f1s,  # Add F1 history
        'val_aucs': val_aucs,  # Add AUROC history
        'best_val_acc': best_val_acc,
        'best_val_f1': best_val_f1,  # Add best F1
        'best_val_auc': best_val_auc,  # Add best AUROC
        'epochs_trained': len(train_losses)
    }
    
    return model, history


def evaluate_model(model, iterator, criterion, device, model_name, num_classes):
    """Evaluate model on test set and return metrics"""
    model.eval()
    test_loss = 0
    test_preds = []
    test_labels = []
    test_probs = []
    
    with torch.no_grad():
        for batch in iterator:
            text, text_lengths, labels = process_batch(batch, debug=False)
            predictions = model(text, text_lengths)
            loss = criterion(predictions, labels)
            test_loss += loss.item()
            
            probs = torch.softmax(predictions, dim=1)
            preds = torch.argmax(predictions, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
    
    avg_test_loss = test_loss / len(iterator)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='weighted')
    
    # Calculate AUC-ROC
    try:
        test_probs_array = np.array(test_probs)
        test_labels_bin = label_binarize(test_labels, classes=range(num_classes))
        test_auc = roc_auc_score(test_labels_bin, test_probs_array, average='weighted', multi_class='ovr')
    except Exception as e:
        print(f"Warning: Could not calculate AUC-ROC for {model_name}: {e}")
        test_auc = 0.0
    
    return avg_test_loss, test_acc, test_f1, test_auc