import torch

dropout_rate = 0.5

max_word_len = 20

batch_size = 10     # default: 10

seq_len = 50        # default: 50

epochs = 50         # default: 50

initial_lr = 0.015  # default: 0.015

decay_rate = 0.1

momentum = 0.9

clipping_value = 5.0

token_emb_size = 100

word_emb_size = 30

padding = "left"    # right or center

break_simmetry = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
