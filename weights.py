import torch, random, os
import numpy as np

fasttext = torch.load('embedding_weights_fasttext.pt', weights_only=False)
unk = torch.load('embedding_weights_unk.pt', weights_only=False)
sec = torch.load('embedding_weights_secondary.pt', weights_only=False)


print(fasttext.weight[14])
# print(unk.weight[0])