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
        df = pd.read_csv(filename, sep=" ", skip_blank_lines=True)

        for token in df["-TOKEN-"]:
            if len(str(token)) > 0: # adds empty string????
                token_voc.add(preprocess(token))

                for symbol in str(token):
                    char_voc.add(symbol)

        for tag in df["-NETAG-"]:
            if type(tag) == str:
                tag_voc.add(tag)

        if "" in token_voc:
            token_voc.remove("")

    return (token_voc, char_voc, tag_voc)

def get_labeled_tokens(filename):
    df = pd.read_csv(filename, sep=" ", skip_blank_lines=True)
    tokens = []
    for token in df["-TOKEN-"]:
        if len(str(token)) > 0 and token != "":
            tokens.append(resize(token))
    
    labels = []
    for label in df["-NETAG-"]:
        if type(label) == str:
            labels.append(label)

    return (tokens, labels)

if __name__ == '__main__':
    limit_memory(7 * 1024 * 1024 * 1024)
    token_voc, char_voc, tag_voc = unpack(["conll2003/test.txt", "conll2003/train.txt", "conll2003/valid.txt"])

    train_tokens, train_labels = get_labeled_tokens("conll2003/train.txt")
    print("\nTotal train tokens: " + str(len(train_tokens)))
    save_obj(train_tokens, "train_tokens")
    save_obj(train_labels, "train_labels")

    val_tokens, val_labels = get_labeled_tokens("conll2003/valid.txt")
    print("Total val tokens: " + str(len(val_tokens)))
    save_obj(val_tokens, "val_tokens")
    save_obj(val_labels, "val_labels")

    test_tokens, test_labels = get_labeled_tokens("conll2003/test.txt")
    print("Total test tokens: " + str(len(test_tokens)))
    save_obj(test_tokens, "test_tokens")
    save_obj(test_labels, "test_labels")

    mini_tokens, mini_labels = get_labeled_tokens("conll2003/mini.txt")
    print("Total mini tokens: " + str(len(mini_tokens)))
    save_obj(mini_tokens, "mini_tokens")
    save_obj(mini_labels, "mini_labels")

    print("\nToken voc len: " + str(len(token_voc)))
    print("Char voc len: " + str(len(char_voc)))
    print("Tag voc len: " + str(len(tag_voc)))

    save_obj(token_voc, "token_voc")
    save_obj(char_voc, "char_voc")
    save_obj(tag_voc, "tag_voc")

    char_to_idx = {char: idx for idx, char in enumerate(char_voc)}
    char_to_idx["<pad>"] = len(char_to_idx)

    tag_to_idx = {tag: idx for idx, tag in enumerate(tag_voc)}
    idx_to_tag = {tag_to_idx[tag]: tag for tag in tag_voc}

    tok_to_idx = {tok: idx for idx, tok in enumerate(token_voc)}
    idx_to_tok = {tok_to_idx[tok]: tok for tok in token_voc}

    save_obj(char_to_idx, "char_to_idx")
    save_obj(tag_to_idx, "tag_to_idx")
    save_obj(idx_to_tag, "idx_to_tag")
    save_obj(tok_to_idx, "tok_to_idx")
    save_obj(idx_to_tok, "idx_to_tok")

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