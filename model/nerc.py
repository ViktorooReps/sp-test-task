from torch import nn
import torch
import random

import numpy as np

from torchcrf import CRF
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from pprint import pprint


class PaddingCollator:
    """Pads tokens with char_pad to max_word_len and tokens
    with tok_pad to the maximum length of batch's sequence"""

    def __init__(self, char_pad, max_word_len, tok_pad, tag_pad):
        self.char_pad = char_pad
        self.tok_pad = tok_pad
        self.tag_pad = tag_pad
        self.max_word_len = max_word_len

    def pad_token(self, tok_chars):
        num_pads = self.max_word_len - len(tok_chars)
        return tok_chars + num_pads * [self.char_pad]
    
    def __call__(self, batch):
        char_3d_idxs = [
            [self.pad_token(token_chars) for token_chars in trio[0]]
            for trio in batch
        ]
        tok_2d_idxs = [trio[1] for trio in batch]
        lbl_2d_idxs = [trio[2] for trio in batch]

        seq_lengths = [len(seq) for seq in tok_2d_idxs]
        max_seq_len = max(seq_lengths)

        for seq_idx, curr_seq_len in enumerate(seq_lengths):
            num_pads = max_seq_len - curr_seq_len
            tok_2d_idxs[seq_idx] += num_pads * [self.tok_pad]
            lbl_2d_idxs[seq_idx] += num_pads * [self.tag_pad]
            char_3d_idxs[seq_idx] += num_pads * [[self.char_pad] * self.max_word_len]

        return (
            torch.tensor(char_3d_idxs, requires_grad=False),
            torch.tensor(tok_2d_idxs, requires_grad=False),
            torch.tensor(lbl_2d_idxs, requires_grad=False), 
            seq_lengths
        )


def get_train_dataloader(data, batch_size, pad_collator, worker_init_fn):
    return DataLoader(
        data, batch_size,  
        collate_fn=pad_collator,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        sampler=RandomSampler(data)
    )

def get_eval_dataloader(data, batch_size, pad_collator, worker_init_fn):
    return DataLoader(
        data, batch_size,  
        collate_fn=pad_collator,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        sampler=SequentialSampler(data)
    )


class Data(Dataset):
    """Stores sequences as list of pairs (tokens, tags)"""

    SEQS_TOKS = 0
    SEQS_TAGS = 1

    def __init__(self, seqs, char_to_idx, tok_to_idx, tag_to_idx, aug=False, 
        max_token_len=20, preprocessor=None):
        self.seqs = seqs
        self.aug = aug
        self.max_token_len = max_token_len
        self.char_to_idx = char_to_idx
        self.tok_to_idx = tok_to_idx
        self.tag_to_idx = tag_to_idx

        if preprocessor == None:
            self.preprocessor = lambda x: x
        else:
            self.preprocessor = preprocessor

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        toks = [tok for tok in self.seqs[idx][self.SEQS_TOKS]]
        tags = [tag for tag in self.seqs[idx][self.SEQS_TAGS]]
        chars = [[char for char in tok] for tok in toks]

        toks = [self.tok_to_idx[self.preprocessor(tok)] for tok in toks]
        tags = [self.tag_to_idx[tag] for tag in tags]
        chars = [
            [self.char_to_idx[char] for char in char_lst]
            for char_lst in chars
        ]

        return chars, toks, tags


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

        self.char_dropout = nn.Dropout(p=dropout_rate)

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

    def forward(self, chars, toks, seq_lengths):
        """
        chars: padded indicies of chars in tokens
        toks: padded to max seq len tokens
        """
        xc = self.char_embs(chars)
        xc = self.char_dropout(xc)
        xc = torch.unsqueeze(xc, 1)
        xc = self.conv(xc)
        xc = self.max_pool(xc)
        xc = torch.squeeze(xc)
        xc = xc.permute(0, 2, 1)

        xt = self.tok_embs(toks)

        x = torch.cat([xt, xc], dim=2)
        x = self.inp_dropout(x)
        x = pack_padded_sequence(x, seq_lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x) 
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.outp_dropout(x)
        x = self.hidden2emissions(x)

        return x

    def init_embeddings(self, num_chars, char_emb_size, voc_size, token_emb_size, token_vecs):
        self.char_embs = nn.Embedding(num_chars, char_emb_size)
        self.tok_embs = nn.Embedding(voc_size, token_emb_size)

        nn.init.kaiming_uniform_(self.char_embs.weight)
        self.tok_embs.load_state_dict({"weight": token_vecs})

    def build_mask(self, seq_lens):
        """Converts seq_lens list to mask for CRF"""

        max_seq_len = max(seq_lens)
        return torch.stack([
                torch.cat(
                    [
                        torch.ones(seq_len, dtype=torch.bool),
                        torch.zeros(max_seq_len - seq_len, dtype=torch.bool)
                    ],
                    dim=0
                ) for seq_len in seq_lens
            ])

    def loss(self, emissions, labels, seq_lens):
        return - self.crf(emissions, labels, mask=self.build_mask(seq_lens), reduction="mean")

    def decode(self, emissions, seq_lens):
        """Returns unpadded predicted tags"""

        return self.crf.decode(emissions, mask=self.build_mask(seq_lens))

