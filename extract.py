import pandas as pd
import numpy as np

import torch
import csv
import re

from model.hyperparams import max_word_len
from utils.memory_management import limit_memory, load_obj, save_obj

def preprocess(token):
    return str(resize(token)).lower()

def resize(token):
    token = str(token)
    if len(token) > max_word_len:
        new_token = max(re.split('\W+', token), key=lambda x: len(x))
        if new_token == "":
            new_token = token[:max_word_len]

        return new_token
    else:
        return token

def unpack(filenames):
    token_voc = set()
    char_voc = set()
    tag_voc = set()

    for filename in filenames:
        print("Unpacking " + filename)

        with open(filename) as f:
            header = f.readline() 

            for line in f:
                if line != "\n":
                    token, _, _, label = line.rstrip().split(" ")
                    token_voc.add(preprocess(token))
                    tag_voc.add(label)
                    for char in token:
                        char_voc.add(char)

    return (token_voc, char_voc, tag_voc)

def extract_seqs(filename):
    with open(filename) as f:
        file_contents = f.read() 
    
    seqs = [
        get_labeled_tokens(seq)
        for seq in file_contents.split("\n\n")
    ]

    return seqs

def get_labeled_tokens(seq):
    tokens = []
    labels = []

    for line in seq.split("\n"):
        token, _, _, label = line.rstrip().split(" ")
        tokens.append(resize(token))
        labels.append(label)

    return (tokens, labels)

if __name__ == '__main__':
    limit_memory(7 * 1024 * 1024 * 1024)
    token_voc, char_voc, tag_voc = unpack(["conll2003/test.txt", "conll2003/train.txt", "conll2003/valid.txt"])
    token_voc.add("<pad>")
    char_voc.add("<pad>")

    print("\nToken voc len: " + str(len(token_voc)))
    print("Char voc len: " + str(len(char_voc)))
    print("Tag voc len: " + str(len(tag_voc)))

    save_obj(token_voc, "token_voc")
    save_obj(char_voc, "char_voc")
    save_obj(tag_voc, "tag_voc")

    char_to_idx = {char: idx for idx, char in enumerate(char_voc)}

    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_voc)}
    idx_to_tag = {tag_to_idx[tag]: tag for tag in tag_voc}

    tok_to_idx = {tok: idx for idx, tok in enumerate(token_voc)}
    idx_to_tok = {tok_to_idx[tok]: tok for tok in token_voc}

    save_obj(char_to_idx, "char_to_idx")
    save_obj(tag_to_idx, "tag_to_idx")
    save_obj(idx_to_tag, "idx_to_tag")
    save_obj(tok_to_idx, "tok_to_idx")
    save_obj(idx_to_tok, "idx_to_tok")

    train_seqs = extract_seqs("conll2003/train.txt")
    print("\nTotal train sequences: " + str(len(train_seqs)))
    save_obj(train_seqs, "train_seqs")

    val_seqs = extract_seqs("conll2003/valid.txt")
    print("Total valid sequences: " + str(len(val_seqs)))
    save_obj(val_seqs, "val_seqs")

    test_seqs = extract_seqs("conll2003/test.txt")
    print("Total test sequences: " + str(len(test_seqs)))
    save_obj(test_seqs, "test_seqs")

    mini_seqs = extract_seqs("conll2003/mini.txt")
    print("Total mini sequences: " + str(len(mini_seqs)))
    save_obj(mini_seqs, "mini_seqs")

    print("\nExtracting token embeddings")
    glove_data = np.loadtxt("glove-embs/glove.6B.100d.txt", dtype='str', comments=None)

    glove_dict = {token : torch.tensor([float(num) for num in emb], requires_grad=True).view(1, -1)
        for token, emb in zip(glove_data[:, 0], glove_data[:, 1:])}

    glove_voc = set(glove_dict.keys())
    print("GloVe voc len: " + str(len(glove_voc)))

    oov_voc = set()
    oov_idxs = set()

    tok_to_emb = {}
    for token in token_voc:
        if token in glove_voc:
            tok_to_emb[token] = glove_dict[token]
        else:
            oov_voc.add(token)
            oov_idxs.add(tok_to_idx[token])

    print("Out of vocabulary tokens: " + str(len(oov_voc)))

    save_obj(oov_voc, "oov_voc")
    save_obj(oov_idxs, "oov_idxs")
    save_obj(tok_to_emb, "tok_to_emb")