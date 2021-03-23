from torch import nn
import torch

import numpy as np

from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader

def collate_fn(batch):
    return (
        torch.tensor([sample[0] for sample in batch], requires_grad=False, dtype=torch.int64),
        torch.tensor([sample[1] for sample in batch], requires_grad=False, dtype=torch.int64),
        torch.tensor([sample[2] for sample in batch], requires_grad=False, dtype=torch.int64),
    )

class Data(Dataset):

    def __init__(self, X, y, tok_to_idx, char_to_idx, tag_to_idx, aug=False, max_token_len=20, padding="left"):
        self.X = X 
        self.y = y
        self.aug = aug
        self.tok_to_idx = tok_to_idx
        self.char_to_idx = char_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_token_len = max_token_len
        self.padding = padding

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        token = self.X[idx]
        tok_idx = self.tok_to_idx[token]
        lbl_idx = self.tag_to_idx[self.y[idx]]
        char_idxs = [self.char_to_idx[char] for char in token]

        pad_idx = self.char_to_idx["<pad>"]
        num_pads = self.max_token_len - len(char_idxs)

        if self.padding == "center":
            left_pads = num_pads // 2
            right_pads = num_pads // 2 + num_pads % 2

        if self.padding == "left":
            left_pads = 0
            right_pads = num_pads

        if self.padding == "left":
            left_pads = num_pads
            right_pads = 0

        char_idxs = [pad_idx] * left_pads + char_idxs + [pad_idx] * right_pads

        return (char_idxs, tok_idx, lbl_idx)


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
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=60, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20, sent_len=10, dropout_rate=0.5):
        super(CNNbLSTMCRF, self).__init__()

        self.init_embeddings(len(char_to_idx), char_emb_size, len(tok_to_idx), token_emb_size, token_vecs)

        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=char_repr_size, 
                              kernel_size=(3, char_emb_size),
                              padding=(1, 0))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, max_word_len))

        self.inp_dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(token_emb_size + char_repr_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.outp_dropout = nn.Dropout(p=dropout_rate)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                nn.init.zeros_(bias)

                # setting forget gates biases to 1.0
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                nn.init.xavier_uniform_(weight)

        self.hidden2emissions = nn.Linear(lstm_hidden_size * 2, len(tag_to_idx))
        nn.init.xavier_uniform_(self.hidden2emissions.weight)
        nn.init.zeros_(self.hidden2emissions.bias)

        self.crf = CRF(len(tag_to_idx), batch_first=True)

    def forward(self, chars, toks):
        """
        chars: (batch_size, num_chars) 
        toks: (batch_size)
        """
        xc = self.char_embs(chars)
        xc = torch.unsqueeze(xc, 1)
        xc = self.conv(xc)
        xc = torch.squeeze(xc)
        xc = self.max_pool(xc)
        xc = torch.squeeze(xc)

        xt = self.tok_embs(toks)

        x = torch.cat([xt, xc], dim=1)
        x = torch.unsqueeze(x, 0)
        x = self.inp_dropout(x)
        x, _ = self.lstm(x)
        x = self.outp_dropout(x)
        x = self.hidden2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def loss(self, emissions, labels):
        return - self.crf(emissions, labels)

    def decode(self, emissions):
        return self.crf.decode(emissions)


class CNNCRF(nn.Module):
    """
    Subnet of CNNbLSTMCRF
    """
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=30, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20, sent_len=10, dropout_rate=0.5):
        super(CNNCRF, self).__init__()

        self.init_embeddings(len(char_to_idx), char_emb_size, len(tok_to_idx), token_emb_size, token_vecs)

        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=char_repr_size, 
                              kernel_size=(3, char_emb_size),
                              padding=(1, 0))
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.max_pool = nn.MaxPool2d(kernel_size=(1, max_word_len))

        self.dropout = nn.Dropout(p=dropout_rate)

        self.hidden2emissions = nn.Linear(char_repr_size + token_emb_size, len(tag_to_idx))
        nn.init.xavier_uniform_(self.hidden2emissions.weight)
        nn.init.zeros_(self.hidden2emissions.bias)

        self.crf = CRF(len(tag_to_idx), batch_first=True)

    def forward(self, chars, toks):
        """
        chars: (batch_size, num_chars) 
        toks: (batch_size)
        """
        xc = self.char_embs(chars)
        xc = torch.unsqueeze(xc, 1)
        xc = self.conv(xc)
        xc = torch.squeeze(xc)
        xc = self.max_pool(xc)
        xc = torch.squeeze(xc)

        xt = self.tok_embs(toks)

        x = torch.cat([xt, xc], dim=1)
        x = torch.unsqueeze(x, 0)
        x = self.dropout(x)
        x = self.hidden2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def loss(self, emissions, labels):
        return - self.crf(emissions, labels)

    def decode(self, emissions):
        return self.crf.decode(emissions)


class bLSTMCRF(nn.Module):
    """
    Subnet of CNNbLSTMCRF
    """
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=30, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20, sent_len=10, dropout_rate=0.5):
        super(bLSTMCRF, self).__init__()

        self.init_embeddings(len(char_to_idx), char_emb_size, len(tok_to_idx), token_emb_size, token_vecs)

        self.inp_dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(token_emb_size, lstm_hidden_size, bidirectional=True, batch_first=True)
        self.outp_dropout = nn.Dropout(p=dropout_rate)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                nn.init.zeros_(bias)

                # setting forget gates biases to 1.0
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

            for name in filter(lambda n: "weight" in n,  names):
                weight = getattr(self.lstm, name)
                nn.init.xavier_uniform_(weight)

        self.hidden2emissions = nn.Linear(lstm_hidden_size * 2, len(tag_to_idx))
        nn.init.xavier_uniform_(self.hidden2emissions.weight)
        nn.init.zeros_(self.hidden2emissions.bias)

        self.crf = CRF(len(tag_to_idx), batch_first=True)

    def forward(self, chars, toks):
        """
        chars: (batch_size, num_chars) 
        toks: (batch_size)
        """
        x = self.tok_embs(toks)
        x = torch.unsqueeze(x, 0)

        x = self.inp_dropout(x)
        x, _ = self.lstm(x)
        x = self.outp_dropout(x)
        x = self.hidden2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def loss(self, emissions, labels):
        return - self.crf(emissions, labels)

    def decode(self, emissions):
        return self.crf.decode(emissions)


class OnlyCRF(nn.Module):
    """
    Subnet of CNNbLSTMCRF
    """
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=30, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20, sent_len=10, dropout_rate=0.5):
        super(OnlyCRF, self).__init__()

        self.init_embeddings(len(char_to_idx), char_emb_size, len(tok_to_idx), token_emb_size, token_vecs)

        self.embs2emissions = nn.Linear(token_emb_size, len(tag_to_idx))
        nn.init.xavier_uniform_(self.embs2emissions.weight)
        nn.init.zeros_(self.embs2emissions.bias)

        self.crf = CRF(len(tag_to_idx), batch_first=True)

    def forward(self, chars, toks):
        """
        chars: (batch_size, num_chars) 
        toks: (batch_size)
        """
        x = self.tok_embs(toks)
        x = torch.unsqueeze(x, 0)

        x = self.embs2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def loss(self, emissions, labels):
        return - self.crf(emissions, labels)

    def decode(self, emissions):
        return self.crf.decode(emissions)