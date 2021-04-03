import torch

# model init

dropout_rate = 0.5

max_word_len = 20

batch_size = 10     # default: 10

epochs = 65         # default: 50

initial_lr = 0.015  # default: 0.015

decay_rate = 0.05

momentum = 0.9

clipping_value = 5.0

token_emb_size = 100

word_emb_size = 30

break_simmetry = True

# active learning

starting_size = 500

request_seqs = 100

model_tolerance = 5

model_min_epochs = None

model_max_epochs = None

global_tolerance = 5

global_min_epochs = 30

global_max_epochs = 30

# misc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
