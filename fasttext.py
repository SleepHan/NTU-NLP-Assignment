import torch, random, os
from torchtext import data , datasets
from collections import defaultdict, Counter
import numpy as np

os.environ['GENSIM_DATA_DIR'] = os.path.join(os.getcwd(), 'gensim-data')

import gensim.downloader as api
from gensim.models import KeyedVectors

### Part 0: Dataset Preparation

# For tokenization
TEXT = data.Field ( tokenize = 'spacy', tokenizer_language = 'en_core_web_sm', include_lengths = True )

# For multi - class classification labels
LABEL = data.LabelField ()

# Load the TREC dataset
# Train / Validation / Test split
train_data, test_data = datasets.TREC.splits( TEXT, LABEL, fine_grained = False )
train_data, validation_data = train_data.split(
    split_ratio=0.8,
    stratified=True,
    strata_field='label',
    random_state=random.seed(42)
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

'''

Label distribution in training set:
- ABBR: 69 samples (1.58%)
- DESC: 930 samples (21.32%)
- ENTY: 1000 samples (22.93%)
- HUM: 978 samples (22.42%)
- LOC: 668 samples (15.31%)
- NUM: 717 samples (16.44%)
Total samples: 4362, Sum of percentages: 100.00%

Might need to apply oversampling techniques for ABBR category during training.
Example: Data/Text Augmentation. (Library: nlpaug) https://nlpaug.readthedocs.io/en/latest/augmenter/word/synonym.html

'''


### Part 1: Preparing Word Embeddings
# Word2Vec Approach

#### a) Size of Vocabulary formed from training data according to tokenization method
# Vocabulary size (includes specials like <unk>, <pad>)
TEXT.build_vocab(train_data, min_freq=1)
vocab_size = len(TEXT.vocab)
print("Vocabulary Size (with specials):", vocab_size)

vocab_wo_specials = len([w for w in TEXT.vocab.stoi if w not in {TEXT.unk_token, TEXT.pad_token}])
print("Vocabulary size (no specials):", vocab_wo_specials)


# #### c) OOV mitigation strategy (No transformer-based language models allowed)
# Implement your solution in your source code. Show the corresponding code snippet.
# 1. Fast Text Model Implementatation
# Load FastText with subword info (pretrained on Wikipedia)
# First download is large; cached afterwards
ft = api.load("fasttext-wiki-news-subwords-300")
# https://fasttext.cc/docs/en/english-vectors.html

embedding_dim = ft.vector_size

# Build embedding matrix aligned to TEXT.vocab
num_tokens = len(TEXT.vocab)
emb_matrix = np.zeros((num_tokens, embedding_dim), dtype=np.float32)

# torchtext 0.4.0: TEXT.vocab.itos is index->token, stoi is token->index
pad_tok = TEXT.pad_token
unk_tok = TEXT.unk_token

for idx, token in enumerate(TEXT.vocab.itos):
    # # FastText can infer OOV vectors via subword decomposition
    # emb_matrix[idx] = ft.get_vector(token)

    # Skip specials here; we will set them explicitly below
    if token in {pad_tok, unk_tok}:
        continue
    try:
        emb_matrix[idx] = ft.get_vector(token)
    except KeyError:
        # Try lowercase variant
        if token.lower() in ft.key_to_index:
            emb_matrix[idx] = ft.get_vector(token.lower())
        else:
            # Fallback to small random init (FastText usually can infer via subwords,
            # but this protects against any residual misses)
            emb_matrix[idx] = np.random.uniform(-0.05, 0.05, embedding_dim).astype(np.float32)

# Create Embedding layer initialized with FastText
embedding = torch.nn.Embedding(num_tokens, embedding_dim, padding_idx=TEXT.vocab.stoi[TEXT.pad_token])
embedding.weight.data.copy_(torch.from_numpy(emb_matrix))

torch.save(embedding, 'embedding_weights_fasttext.pt')

# Optionally freeze during initial training
# embedding.weight.requires_grad = False

# Quick sanity: how many tokens were truly OOV to FastText's main vocab
# (still get vectors via subwords, so this is informational only)
# fasttext_known = sum(1 for t in TEXT.vocab.itos if ft.has_index_for(t))
fasttext_known = 0
words_to_remove = []
for t in TEXT.vocab.itos:
    if t in {pad_tok, unk_tok}:
        continue
    try:
        _ = ft.get_vector(t)
        words_to_remove.append(t)
        fasttext_known += 1
    except KeyError:
        if t.lower() in ft.key_to_index:
            fasttext_known += 1

remaing_words = [word for word in ft.key_to_index if word not in words_to_remove]
unmatched_ft = KeyedVectors(vector_size=ft.vector_size)
unmatched_ft.add_vectors(remaing_words, [ft[w] for w in remaing_words])

print(f"FastText main-vocab known types: {fasttext_known}/{num_tokens} (all tokens still have vectors via subwords)")
print(f'{len(unmatched_ft)}')


# 2. Modelling Unknown (<UNK>) token approach

# Make the <unk> vector informative and trainable by initializing it
# as the mean of available pretrained vectors.
unk_index = TEXT.vocab.stoi[TEXT.unk_token]
known_vecs = []
for token in TEXT.vocab.itos:
    if token in {pad_tok, unk_tok}:
        continue
    try:
        vec = ft.get_vector(token)
        known_vecs.append(vec)
    except KeyError:
        if token.lower() in ft.key_to_index:
            known_vecs.append(ft.get_vector(token.lower()))
        # else: skip

if len(known_vecs) > 0:
    unk_mean = torch.tensor(np.mean(known_vecs, axis=0), dtype=torch.float32)
else:
    unk_mean = torch.empty(embedding_dim).uniform_(-0.05, 0.05)
with torch.no_grad():
    embedding.weight[unk_index] = unk_mean
embedding.weight.requires_grad = True

torch.save(embedding, 'embedding_weights_unk.pt')


# 3. Create a secondary model for OOV words
"""
Strategy summary (1.c):
- Primary mitigation: use FastText subword embeddings to initialize the vocabulary matrix.
  This gives vectors to words unseen in Word2Vec/GloVe via subword composition.
- Strengthen <unk>: initialize as mean FastText vector of known words and keep it trainable.
- Optional fallback: a tiny character-level encoder to synthesize vectors for rare cases
  still missing in FastText's main index, and to allow task-specific adaptation.
"""

from typing import List, Dict


def build_char_vocab(tokens: List[str]) -> Dict[str, int]:
    chars = {"<pad>": 0, "<unk>": 1}
    nxt = 2
    for tok in tokens:
        for ch in tok:
            if ch not in chars:
                chars[ch] = nxt
                nxt += 1
    return chars


class CharBiGRUEncoder(torch.nn.Module):
    def __init__(self, vocab_size: int, char_emb_dim: int, hidden_dim: int, out_dim: int, pad_idx: int = 0):
        super().__init__()
        self.char_embed = torch.nn.Embedding(vocab_size, char_emb_dim, padding_idx=pad_idx)
        self.bigru = torch.nn.GRU(char_emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.proj = torch.nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, char_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.char_embed(char_ids)
        packed = torch.nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.bigru(packed)  # [2, B, H]
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # [B, 2H]
        return self.proj(h)  # [B, out_dim]


def words_to_char_batch(words: List[str], char2idx: Dict[str, int], device: torch.device):
    pad_idx = char2idx["<pad>"]
    unk_idx = char2idx["<unk>"]
    lengths = torch.tensor([len(w) for w in words], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(words) else 0
    ids = torch.full((len(words), max_len), pad_idx, dtype=torch.long)
    for i, w in enumerate(words):
        for j, ch in enumerate(w[:max_len]):
            ids[i, j] = char2idx.get(ch, unk_idx)
    return ids.to(device), lengths.to(device)


# Build a char vocab once from training tokens
all_train_tokens = []
for ex in train_data.examples:
    all_train_tokens.extend(ex.text)
char2idx = build_char_vocab(all_train_tokens)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
char_encoder = CharBiGRUEncoder(
    vocab_size=len(char2idx),
    char_emb_dim=64,
    hidden_dim=128,
    out_dim=embedding_dim,
    pad_idx=char2idx["<pad>"]
).to(device)


def synthesize_vectors_for_tokens(tokens: List[str]) -> torch.Tensor:
    if len(tokens) == 0:
        return torch.empty(0, embedding_dim, device=device)
    char_ids, lengths = words_to_char_batch(tokens, char2idx, device)
    return char_encoder(char_ids, lengths)


# Example of using the fallback for tokens not in FastText's main index
# (FastText usually covers OOV via subwords; this is an extra safety net.)
vocab_tokens = TEXT.vocab.itos
oov_ft_tokens = [t for t in vocab_tokens if t not in {pad_tok, unk_tok} and t.lower() not in ft.key_to_index and t not in ft.key_to_index]
print(f"Tokens missing in FastText main index (before subword): {len(oov_ft_tokens)}")
oov_vecs = synthesize_vectors_for_tokens(oov_ft_tokens)
# You may replace their rows in `embedding.weight` or keep these vectors
# and blend during the model forward pass.

torch.save(oov_vecs, 'embedding_weights_secondary.pt')