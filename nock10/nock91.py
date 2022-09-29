from nock90 import train_dataloader, val_dataloader, tr_dexj, tr_dexe, te_dexj, te_dexe, dic_j, dic_e
from curses import use_env
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import numpy as np
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
import numpy as np
import transformers
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch import cuda
from torch import Tensor
from torch.nn import Transformer
import math
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(DEVICE))

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):#単語の位置情報の埋め込み
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == 1).transpose(0, 1)
    tgt_padding_mask = (tgt == 1).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(dic_j)
TGT_VOCAB_SIZE = len(dic_e)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=1)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
    
        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()


    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)


from timeit import default_timer as timer
NUM_EPOCHS = 12

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer.state_dict(), "/home2/y2019/o1910142/use_model.pth")

"""
Using cuda device
Epoch: 1, Train loss: 5.356, Val loss: 4.883, Epoch time = 1648.470s
Epoch: 2, Train loss: 4.223, Val loss: 4.270, Epoch time = 1656.270s
Epoch: 3, Train loss: 3.734, Val loss: 3.964, Epoch time = 1655.097s
Epoch: 4, Train loss: 3.420, Val loss: 3.803, Epoch time = 1651.114s
Epoch: 5, Train loss: 3.194, Val loss: 3.706, Epoch time = 1653.548s
Epoch: 6, Train loss: 3.017, Val loss: 3.700, Epoch time = 1651.900s
Epoch: 7, Train loss: 2.868, Val loss: 3.638, Epoch time = 1653.265s
Epoch: 8, Train loss: 2.738, Val loss: 3.697, Epoch time = 1662.157s
Epoch: 9, Train loss: 2.628, Val loss: 3.677, Epoch time = 1652.935s
Epoch: 10, Train loss: 2.533, Val loss: 3.720, Epoch time = 1669.608s
Epoch: 11, Train loss: 2.453, Val loss: 3.739, Epoch time = 1663.287s
Epoch: 12, Train loss: 2.394, Val loss: 3.835, Epoch time = 1659.974s
Epoch: 13, Train loss: 2.363, Val loss: 3.827, Epoch time = 1653.537s
Epoch: 14, Train loss: 2.321, Val loss: 3.882, Epoch time = 1659.309s
Epoch: 15, Train loss: 2.274, Val loss: 3.949, Epoch time = 1659.135s
Epoch: 16, Train loss: 2.229, Val loss: 3.970, Epoch time = 1656.585s
Epoch: 17, Train loss: 2.188, Val loss: 3.967, Epoch time = 1657.210s
Epoch: 18, Train loss: 2.151, Val loss: 4.080, Epoch time = 1653.930s
Epoch: 19, Train loss: 2.116, Val loss: 4.073, Epoch time = 1654.082s
Epoch: 20, Train loss: 2.085, Val loss: 4.155, Epoch time = 1662.271s
Epoch: 21, Train loss: 2.056, Val loss: 4.188, Epoch time = 1674.422s
Epoch: 22, Train loss: 2.030, Val loss: 4.185, Epoch time = 1676.472s
Epoch: 23, Train loss: 2.004, Val loss: 4.219, Epoch time = 1675.405s
Epoch: 24, Train loss: 1.981, Val loss: 4.279, Epoch time = 1676.138s
Epoch: 25, Train loss: 1.959, Val loss: 4.283, Epoch time = 1660.342s
Epoch: 26, Train loss: 1.939, Val loss: 4.310, Epoch time = 1651.977s
Epoch: 27, Train loss: 1.920, Val loss: 4.362, Epoch time = 1649.328s
Epoch: 28, Train loss: 1.901, Val loss: 4.366, Epoch time = 1651.929s
Epoch: 29, Train loss: 1.885, Val loss: 4.442, Epoch time = 1649.464s
Epoch: 30, Train loss: 1.867, Val loss: 4.428, Epoch time = 1653.875s

Epoch: 1, Train loss: 5.356, Val loss: 4.883, Epoch time = 1642.030s
Epoch: 2, Train loss: 4.223, Val loss: 4.270, Epoch time = 1651.759s
Epoch: 3, Train loss: 3.734, Val loss: 3.964, Epoch time = 1648.313s
Epoch: 4, Train loss: 3.420, Val loss: 3.803, Epoch time = 1659.396s
Epoch: 5, Train loss: 3.194, Val loss: 3.706, Epoch time = 1667.914s
Epoch: 6, Train loss: 3.017, Val loss: 3.700, Epoch time = 1652.518s
Epoch: 7, Train loss: 2.868, Val loss: 3.638, Epoch time = 1653.101s
Epoch: 8, Train loss: 2.738, Val loss: 3.697, Epoch time = 1651.114s
Epoch: 9, Train loss: 2.628, Val loss: 3.677, Epoch time = 1670.873s
Epoch: 10, Train loss: 2.533, Val loss: 3.720, Epoch time = 1657.723s
Epoch: 11, Train loss: 2.453, Val loss: 3.739, Epoch time = 1667.991s

nvidia-smi
nohup python -u nock91.py >& /home2/y2019/o1910142/nock91.txt
"""