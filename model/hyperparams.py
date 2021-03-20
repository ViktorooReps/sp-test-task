import torch

max_word_len = 20
batch_size = 10
epochs = 50
gradient_clipping = 5.0
initial_lr = 0.015
decay_rate = 0.05
momentum = 0.9
clipping_value = 5.0

token_emb_size = 100
word_emb_size = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")