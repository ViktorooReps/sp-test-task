import pandas as pd
import numpy as np

import torch
import pickle
import csv
import re

from model.hyperparams import max_word_len

def save_obj(obj, name): 
    with open('pickled/'+ name + '.pkl', 'wb') as f:  
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL) 

def load_obj(name): 
    with open('pickled/' + name + '.pkl', 'rb') as f: 
        return pickle.load(f)

def preprocess(raw_token):
    raw_token = str(raw_token)
    if len(raw_token) > max_word_len:
        raw_token = max(re.split('\W+', raw_token), key=lambda x: len(x))

    return str(raw_token).lower()

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

if __name__ == '__main__':
    token_voc, char_voc, tag_voc = unpack(["conll2003/test.txt", "conll2003/train.txt", "conll2003/valid.txt"])

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