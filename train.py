from extract import load_obj
from model.nerc import CNNbLSTMCRF

import torch

char_to_idx = load_obj("char_to_idx")
tag_to_idx = load_obj("tag_to_idx")
tok_to_idx = load_obj("tok_to_idx")
idx_to_tok = load_obj("idx_to_tok")

token_voc = load_obj("token_voc")

tok_to_emb = load_obj("tok_to_emb")
oov_voc = load_obj("oov_voc")

token_vecs = []
for idx in range(len(tok_to_idx)):
    token = idx_to_tok[idx]
    if token not in oov_voc:
        token_vecs.append(tok_to_emb[token])
    else:
        token_vecs.append(torch.zeros((1, 100)))

token_vecs = torch.cat(token_vecs, dim=0)

model = CNNbLSTMCRF(char_to_idx, tok_to_idx, tag_to_idx, token_vecs)

print("Trainable weights:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name, param.data.shape)
        #print(name, param.data)