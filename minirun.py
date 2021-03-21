from utils.memory_management import limit_memory, load_obj, save_obj
from model.nerc import CNNbLSTMCRF, Data, collate_fn

from math import sqrt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.hyperparams import *
from evaluate import evaluate_model

import torch
import torch.optim as optim
import datetime

def train_epoch(model, dataloader, scheduler, optimizer):
    model.train()
    for batch_idx, (chars, toks, lbls) in enumerate(dataloader):
        chars = chars.to(device)
        toks = toks.to(device)
        lbls = lbls.to(device)

        emissions = model(chars, toks).to(device)
        loss = model.loss(emissions, lbls.reshape(1, batch_size))

        loss.backward()

        clip_grad_norm_(model.parameters(), clipping_value)

        optimizer.step()
        optimizer.zero_grad()
    
    scheduler.step()

if __name__ == '__main__':
    limit_memory(7 * 1024 * 1024 * 1024)

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
            r1 = - 1 / sqrt(token_emb_size)
            r2 = 1 / sqrt(token_emb_size)
            token_vecs.append((r1 - r2) * torch.rand(1, token_emb_size) + r2)

    token_vecs = torch.cat(token_vecs, dim=0)

    model = CNNbLSTMCRF(char_to_idx, tok_to_idx, tag_to_idx, token_vecs)

    print("Trainable weights:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data.shape)
            #print(name, param.data)

    mini_tokens = load_obj("mini_tokens")
    mini_labels = load_obj("mini_labels")
    mini_data = Data(mini_tokens, mini_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    mini_dataloader = DataLoader(
        mini_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    lr_lambda = lambda x: 1 / (1 + decay_rate * x)
    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    print("\nTraining start")
    print(datetime.datetime.now())

    mini_loss, mini_f1 = evaluate_model(model, mini_dataloader)

    print("Pre train results:")
    print("[mini] loss: " + str(mini_loss) + " F1: " + str(mini_f1))

    for epoch in range(1000):
        print("\nTraining epoch " + str(epoch))
        train_epoch(model, mini_dataloader, scheduler, optimizer)

        mini_loss, mini_f1 = evaluate_model(model, mini_dataloader)

        print("[mini] loss: " + str(mini_loss) + " F1: " + str(mini_f1))

    print("\nPost train results:")
    mini_loss, mini_f1 = evaluate_model(model, mini_dataloader)
    print("[mini] loss: " + str(mini_loss) + " F1: " + str(mini_f1))
