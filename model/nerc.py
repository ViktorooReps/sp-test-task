from torch import nn
import torch

import numpy as np

from torchcrf import CRF
from torch.utils.data import torch_data


class Data(torch_data.Dataset):

    def __init__(self, X, y, tok_to_idx, char_to_idx, aug=False, max_token_len=20):
        self.X = X 
        self.y = y
        self.aug = aug
        self.tok_to_idx = tok_to_idx
        self.char_to_idx = char_to_idx
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.X)

    def __get_item__(self, idx):
        token = self.X[idx]
        tok_idx = self.tok_to_idx[token]
        char_idxs = [self.char_to_idx[char] for char in token]

        pad_idx = self.char_to_idx["<pad>"]
        num_pads = self.max_token_len - len(char_idxs)
        left_pads = num_pads // 2
        right_pads = num_pads // 2 + num_pads % 2

        char_idxs = [pad_idx] * left_pads + char_idxs + [pad_idx] * right_pads

        return (char_idxs, tok_idx, self.Y[idx])


class CNNbLSTMCRF(nn.Module):
    """
    CNN:
    (batch_size, 1, char_emb_size, max_word_len)
     || convolution
     \/
    (batch_size, char_repr_size, 1, max_word_len)
     || max pooling
     \/
    (batch_size, char_repr_size, 1, 1)
    """
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=30, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20):
        super(CNNbLSTMCRF, self).__init__()

        self.init_embeddings(len(char_to_idx), char_emb_size, len(tok_to_idx), token_emb_size, token_vecs)

        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=char_repr_size, 
                              kernel_size=(char_emb_size, 3),
                              padding=(0, 1))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, max_word_len))

        self.lstm = nn.LSTM(token_emb_size + char_repr_size, lstm_hidden_size, bidirectional=True)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                nn.init.zeros_(bias)

                # setting forget gates biases to 1.0
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)

            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                nn.init.xavier_uniform_(weight)

        self.crf = CRF(len(tag_to_idx), batch_first=True)


    def forward(self, inputs):
        """Inputs: batches, each batch consists of sequence. 
        Sequence is a list of (char_idxs, tok_idx)"""
        pass

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size) 

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def adapt_to_oov(self, tok_idx_seq, oov_idxs):
        """Uses tokens from [i - wind, i + wind] to estimate OOV's embedding"""
        wind = 4
        with torch.no_grad():
            for i, tok_idx in filter(lambda pair: pair[1] in oov_idxs, enumerate(tok_idx_seq)):
                num = 0
                for wind_i in range(i - wind, i + wind + 1):
                    if wind_i > 0 and wind_i < len(tok_idx_seq) and not wind_i in oov_idxs:
                        num += 1
                        self.tok_embs.weight[i] += self.tok_embs.weight[wind_i]

                self.tok_embs.weight[i] /= num + 1

