from extract import load_obj, save_obj
from utils.memory_management import limit_memory
from model.nerc import CNNbLSTMCRF, Data, collate_fn

from math import sqrt
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from model.hyperparams import *
from evaluate import evaluate_model

import torch
import torch.optim as optim
import time
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

    train_tokens = load_obj("train_tokens")
    train_labels = load_obj("train_labels")
    train_data = Data(train_tokens, train_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    val_tokens = load_obj("val_tokens")
    val_labels = load_obj("val_labels")
    val_data = Data(val_tokens, val_labels, 
        tok_to_idx, char_to_idx, tag_to_idx, max_token_len=max_word_len)

    train_dataloader = DataLoader(
        train_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_data, batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        drop_last=True
    )

    optimizer = optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda epoch: initial_lr / (1 + decay_rate * epoch)
    )

    print("\nTraining start")
    print(datetime.datetime.now())

    train_loss_list = []
    val_loss_list = []

    train_f1_list = []
    val_f1_list = []

    for epoch in range(epochs):
        print("\nTraining epoch " + str(epoch))
        train_epoch(model, train_dataloader, scheduler, optimizer)

        train_loss, train_f1 = evaluate_model(model, train_dataloader)
        val_loss, val_f1 = evaluate_model(model, val_dataloader)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)

        print("[train] loss: " + str(train_loss) + " F1: " + str(train_f1))
        print("[valid] loss: " + str(val_loss) + " F1: " + str(val_f1))

    save_obj(train_loss_list, "train_loss_list")
    save_obj(val_loss_list, "val_loss_list")

    save_obj(train_f1_list, "train_f1_list")
    save_obj(val_f1_list, "val_f1_list")