import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from main import Main
from tokens import Tokens
import torch.distributions as dist
import random, tqdm, sys, math
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence

NUM_TOKENS = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiheadSelfAttention(nn.Module):
    def __init__(self, k, heads = 4, mask = False):
            
        super().__init__()

        assert k % heads == 0, "k must be divisible by heads"

        self.k, self.heads = k, heads

        self.tokeys = nn.Linear(k, k, bias = False)
        self.toqueries = nn.Linear(k, k, bias = False)
        self.tovalues = nn.Linear(k, k, bias = False)

        self.unifyheads = nn.Linear(k, k)

        self.mask = mask

    def forward(self, x):
        B, T, K = x.size()
        H = self.heads

        keys = self.tokeys(x)
        queries = self.toqueries(x)
        values = self.tovalues(x)

        # Use assert to check the shape of the tensors

        S = K // H

        keys = keys.view(B, T, H, S)
        queries = queries.view(B, T, H, S)
        values = values.view(B, T, H, S)

        keys = keys.transpose(1, 2).contiguous().view(B * H, T, S)
        queries = queries.transpose(1, 2).contiguous().view(B * H, T, S)
        values = values.transpose(1, 2).contiguous().view(B * H, T, S)

        dot = torch.bmm(queries, keys.transpose(1, 2))

        # dot = dot / (K ** (1/2))

        # if self.mask:
        #     mask = torch.triu(torch.ones(T, T), diagonal = 1).bool()
        #     mask = mask.unsqueeze(0).unsqueeze(0)
        #     dot = dot.masked_fill(mask, -float('inf'))

        indices = torch.triu_indices(T, T, offset=1)
        dot[:, indices[0], indices[1]] = float('-inf')

        dot = F.softmax(dot, dim = 2)

        # out = torch.bmm(dot, values).view(B, H, T, S)
        # out = out.transpose(1, 2).contiguous().view(B, T, S * H)

        return self.unifyheads(dot)

class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = MultiheadSelfAttention(k, heads=heads)

    self.norm1 = nn.LayerNorm(k)
    self.norm2 = nn.LayerNorm(k)

    self.ff = nn.Sequential(
      nn.Linear(k, 4 * k),
      nn.ReLU(),
      nn.Linear(4 * k, k))

  def forward(self, x):
    attended = self.attention(x)
    x = self.norm1(attended + x)

    fedforward = self.ff(x)
    x = self.norm2(fedforward + x)
    return x
  
class transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_tokens)

def forward(self, x):
    """
    :param x: A (b, t) tensor of integer values representing
                words (in some predetermined vocabulary).
    :return: A (b, c) tensor of log-probabilities over the
                classes (where c is the nr. of classes).
    """
    # generate token embeddings
    tokens = self.token_emb(x)
    b, t, e = tokens.size()

    # generate position embeddings
    positions = torch.arange(t)
    positions = self.pos_emb(torch.arange(t, device='cuda'))[None, :, :].expand(b, t, e)

    x = tokens + positions
    x = self.tblocks(x)

    # Average-pool over the t dimension and project to class
    # probabilities
    x = self.toprobs(x.mean(dim=1))
    return F.log_softmax(x, dim=1)
