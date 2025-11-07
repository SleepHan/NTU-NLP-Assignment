import torch

state = torch.load('weights/rnn_reg_final_best.pt', map_location='cpu')

print("Keys in state dict:")
for k, v in list(state.items())[:15]:
    print(f'{k}: {v.shape}')

print(f'\nEmbedding shape: {state["embedding.weight"].shape}')
print(f'RNN weight_ih shape: {state["rnn.weight_ih_l0"].shape}')
print(f'RNN weight_hh shape: {state["rnn.weight_hh_l0"].shape}')
print(f'FC weight shape: {state["fc.weight"].shape}')
print(f'Has attention: {"attention.weight" in state}')
if "attention.weight" in state:
    print(f'Attention weight shape: {state["attention.weight"].shape}')

# Infer configuration
vocab_size = state["embedding.weight"].shape[0]
hidden_dim = state["rnn.weight_ih_l0"].shape[0]
has_attention = "attention.weight" in state

print(f'\nInferred configuration:')
print(f'  vocab_size: {vocab_size}')
print(f'  hidden_dim: {hidden_dim}')
print(f'  has_attention: {has_attention}')
print(f'  aggregation: {"attention" if has_attention else "unknown"}')




