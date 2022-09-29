import sentencepiece as spm
from sentencepiece import SentencePieceTrainer
from sentencepiece import SentencePieceProcessor
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
from torch.utils.data import DataLoader, TensorDataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List

def sentp(lang):
    spm.SentencePieceTrainer.Train(
        input=f"/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-train.{lang}", 
        model_prefix=f"sentencepiece_nock95tr_{lang}",
        character_coverage=0.9995, #モデルがカバーする語彙割合
        vocab_size=8000,
        pad_id=1,
        bos_id=2,
        eos_id=3,
        unk_piece='<unk>',
        pad_piece='<pad>',
        bos_piece='<bos>',
        eos_piece='<eos>',
    )


sentp("ja")
sentp("en")

def cut(list_j, list_e):
  cut_list_j = []
  cut_list_e = []
  for j, e in zip(list_j, list_e):
    if  len(j) < 30 and len(e) < 30:
      cut_list_j.append(torch.tensor(j))
      cut_list_e.append(torch.tensor(e))    
  return cut_list_j, cut_list_e

def id_change(data, lang):
    sp = SentencePieceProcessor()
    sp.load(f"/home2/y2019/o1910142/sentencepiece_nock95tr_{lang}.model")
    id_l = []
    f = open(f"/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-{data}.{lang}", "r")
    for line in f:
        #l = sp.EncodeAsPieces(line)
        l = sp.EncodeAsIds(line)
        l.append(3)
        l.insert(0, 2)
        id_l.append(l)
    f.close()
    return id_l

tr_dexj = cut(id_change("train", "ja"), id_change("train", "en"))[0]
te_dexj = cut(id_change("dev", "ja"), id_change("dev", "en"))[0]
tr_dexe = cut(id_change("train", "ja"), id_change("train", "en"))[1]
te_dexe = cut(id_change("dev", "ja"), id_change("dev", "en"))[1]

print(len(tr_dexj), len(te_dexj), len(tr_dexe), len(te_dexe))

tr_dexj = pad_sequence(tr_dexj, batch_first=True, padding_value=1)
te_dexj = pad_sequence(te_dexj, batch_first=True, padding_value=1)
tr_dexe = pad_sequence(tr_dexe, batch_first=True, padding_value=1)
te_dexe = pad_sequence(te_dexe, batch_first=True, padding_value=1)


train_ds = TensorDataset(tr_dexj, tr_dexe)
train_dataloader = DataLoader(train_ds, batch_size=64, shuffle=True)

valid_ds = TensorDataset(te_dexj, te_dexe)
val_dataloader = DataLoader(valid_ds, batch_size=64)


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(DEVICE))

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
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

SRC_VOCAB_SIZE = 8000
TGT_VOCAB_SIZE = 8000
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
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
NUM_EPOCHS = 30

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    torch.save(transformer.state_dict(), "/home2/y2019/o1910142/use_model_95.pth")

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == 3:
            break
    return ys

# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    sp = SentencePieceProcessor()
    sp.load(f"/home2/y2019/o1910142/sentencepiece_nock95tr_ja.model")
    id_l = sp.EncodeAsIds(src_sentence)
    id_l.insert(0, 2)
    id_l.append(3)
    src = torch.tensor(id_l).unsqueeze(0).view(-1, 1)
    num_tokens = len(id_l)
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=2).flatten()


    sp1 = SentencePieceProcessor()
    sp1.load(f"/home2/y2019/o1910142/sentencepiece_nock95tr_en.model") 
    ans = sp1.DecodeIds(tgt_tokens.tolist()).replace("<bos>", "").replace("<eos>", "")
    return ans

"""   
print(translate(transformer, "寛元2年（1244年）傘松に大佛寺を開く。"))
    #return tgt_tokens
"""
f = open('/home2/y2019/o1910142/kftt-data-1.0/data/orig/kyoto-dev.ja', 'r')
f1 = open("BLEU_test_95.txt", "w")
for line in f:
    f1.write(translate(transformer, line) + "\n")
    print("f")

f.close()
f1.close()


"""nvidia-smi
nohup python -u nock95.py >& /home2/y2019/o1910142/nock95.txt


Epoch: 1, Train loss: 4.864, Val loss: 4.408, Epoch time = 196.042s
Epoch: 2, Train loss: 3.692, Val loss: 3.616, Epoch time = 196.378s
Epoch: 3, Train loss: 3.105, Val loss: 3.196, Epoch time = 197.257s
Epoch: 4, Train loss: 2.741, Val loss: 2.945, Epoch time = 193.797s
Epoch: 5, Train loss: 2.498, Val loss: 2.823, Epoch time = 195.988s
Epoch: 6, Train loss: 2.320, Val loss: 2.776, Epoch time = 195.326s
Epoch: 7, Train loss: 2.180, Val loss: 2.685, Epoch time = 197.998s
Epoch: 8, Train loss: 2.063, Val loss: 2.660, Epoch time = 198.131s
Epoch: 9, Train loss: 1.964, Val loss: 2.610, Epoch time = 197.254s
Epoch: 10, Train loss: 1.878, Val loss: 2.538, Epoch time = 195.472s
Epoch: 11, Train loss: 1.801, Val loss: 2.574, Epoch time = 197.152s
Epoch: 12, Train loss: 1.733, Val loss: 2.566, Epoch time = 196.048s
"""