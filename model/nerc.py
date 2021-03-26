from torch import nn
import torch
import random

import numpy as np

from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler

from pprint import pprint

def collate_fn(batch):
    """
    Returns:
        tensor of shape (batch_size, seq_len, max_word_len) of char idxs,
        tensor of shape (batch_size, seq_len) of token idxs
        tensor of shape (batch_size, seq_len) of label idxs
    """

    char_3d_idxs = [[tok[0] for tok in seq] for seq in batch]
    tok_2d_idxs = [[tok[1] for tok in seq] for seq in batch]
    lbl_2d_idxs = [[tok[2] for tok in seq] for seq in batch]

    return (
        torch.tensor(char_3d_idxs, requires_grad=False, dtype=torch.int64),
        torch.tensor(tok_2d_idxs, requires_grad=False, dtype=torch.int64),
        torch.tensor(lbl_2d_idxs, requires_grad=False, dtype=torch.int64),
    )

def get_train_dataloader(data, batch_size, seq_len):
    r = random.randrange(0, seq_len)
    return DataLoader(
        data, batch_size,  
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=BatchSampler(
            SequentialSampler(range(r, len(data) - r)),
            batch_size=seq_len,
            drop_last=True
        )
    )

def get_eval_dataloader(data, batch_size, seq_len):
    return DataLoader(
        data, batch_size,  
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
        sampler=BatchSampler(
            SequentialSampler(data),
            batch_size=seq_len,
            drop_last=True
        )
    )


class Data(Dataset):

    def __init__(self, X, y, tok_to_idx, char_to_idx, tag_to_idx, aug=False, 
        max_token_len=20, padding="left", preprocessor=None):
        self.X = X
        self.y = y
        self.aug = aug
        self.tok_to_idx = tok_to_idx
        self.char_to_idx = char_to_idx
        self.tag_to_idx = tag_to_idx
        self.max_token_len = max_token_len
        self.padding = padding

        if preprocessor == None:
            self.preprocessor = lambda x : x
        else:
            self.preprocessor = preprocessor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, seq_idxs):
        """Returns list of (char_idxs, tok_idx, lbl_idx)"""

        seq = []
        for idx in seq_idxs:
            raw_token = self.X[idx]
            token = self.preprocessor(raw_token)
            tok_idx = self.tok_to_idx[token]
            lbl_idx = self.tag_to_idx[self.y[idx]]
            char_idxs = [self.char_to_idx[char] for char in raw_token]

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

            seq.append([char_idxs, tok_idx, lbl_idx])

        return seq


class CNNbLSTMCRF(nn.Module):
    """
    CNN:
    (batch_size, 1, seq_len, max_word_len, char_emb_size)
     || convolution
     \/
    (batch_size, char_repr_size, seq_len, max_word_len, 1)
     || max pooling
     \/
    (batch_size, char_repr_size, seq_len, 1, 1)

    bLSTM:
    (batch_size, seq_len, char_repr_size + token_emb_size)
     ||  
     \/
    (batch_size, seq_len, 2 * hidden_size)

    CRF:
    (batch_size, seq_len, 2 * hidden_size)
     || linear 
     \/
    (batch_size, seq_len, num_tags)
     || CRF
     \/
    (batch_size, seq_len)
    """

    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, 
        char_repr_size=30, token_emb_size=100, lstm_hidden_size=200, max_word_len=20, 
        sent_len=10, dropout_rate=0.5, break_simmetry=True):
        super(CNNbLSTMCRF, self).__init__()

        self.init_embeddings(
            len(char_to_idx), char_emb_size, 
            len(tok_to_idx), token_emb_size, 
            token_vecs
        )

        self.conv = nn.Conv3d(
            in_channels=1, 
            out_channels=char_repr_size, 
            kernel_size=(1, 3, char_emb_size),
            padding=(0, 1, 0)
        )
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

        self.max_pool = nn.MaxPool3d(kernel_size=(1, max_word_len, 1))

        self.inp_dropout = nn.Dropout(p=dropout_rate)
        self.lstm = nn.LSTM(
            token_emb_size + char_repr_size, 
            lstm_hidden_size, 
            bidirectional=True, 
            batch_first=True
        )
        self.outp_dropout = nn.Dropout(p=dropout_rate)

        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                if not break_simmetry:
                    nn.init.zeros_(bias)
                else:
                    torch.nn.init.sparse_(
                        bias.reshape(bias.size()[0], 1), 
                        sparsity=0.5
                    )

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
        xc = self.max_pool(xc)
        xc = torch.squeeze(xc)
        xc = xc.permute(0, 2, 1)

        xt = self.tok_embs(toks)

        x = torch.cat([xt, xc], dim=2)
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
        return - self.crf(emissions, labels, reduction="mean")

    def decode(self, emissions):
        return self.crf.decode(emissions)


class CNNbLSTMSoftmax(nn.Module):
    """
    CNN:
    (batch_size, 1, char_emb_size, max_word_len)
     || convolution
     \/
    (batch_size, char_repr_size, 1, max_word_len)
     || max pooling
     \/
    (batch_size, char_repr_size, 1, 1)

    LSTM:
    (batch_size, char_repr_size + token_emb_size)
     ||  
     \/
    (batch_size, 2 * hidden_size)

    Softmax:
    (batch_size, 2 * hidden_size)
     || softmax 
     \/
    (batch_size)
    """
    def __init__(self, char_to_idx, tok_to_idx, tag_to_idx, token_vecs, char_emb_size=30, char_repr_size=30, 
                 token_emb_size=100, lstm_hidden_size=200, max_word_len=20, sent_len=10, dropout_rate=0.5, 
                 break_simmetry=True):
        super(CNNbLSTMSoftmax, self).__init__()

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
                if not break_simmetry:
                    nn.init.zeros_(bias)
                else:
                    torch.nn.init.sparse_(bias.reshape(bias.size()[0], 1), sparsity=0.5)

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

        self.loss_function = nn.CrossEntropyLoss()

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

        x = torch.squeeze(x)

        x = self.hidden2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def loss(self, emissions, labels):
        return self.loss_function(emissions, labels.view(-1))

    def decode(self, emissions):
        return torch.argmax(emissions, dim=1)


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