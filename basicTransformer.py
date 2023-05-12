import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_TOKENS = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiheadSelfAttention(nn.Module):
    def __init__(self, k, heads, mask = False):
            
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

        S = K // H

        keys = keys.view(B, T, H, S)
        queries = queries.view(B, T, H, S)
        values = values.view(B, T, H, S)

        keys = keys.transpose(1, 2).contiguous().view(B * H, T, S)
        queries = queries.transpose(1, 2).contiguous().view(B * H, T, S)
        values = values.transpose(1, 2).contiguous().view(B * H, T, S)

        dot = torch.bmm(queries, keys.transpose(1, 2))

        dot = dot / (K ** (1/2))

        dot = F.softmax(dot, dim = 2)

        out = torch.bmm(dot, values).view(B, H, T, S)
        out = out.transpose(1, 2).contiguous().view(B, T, S * H)

        return self.unifyheads(out)

class TransformerBlock(nn.Module):
  def __init__(self, k, heads):
    super().__init__()

    self.attention = MultiheadSelfAttention(k = k, heads = heads)

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
  
class basictransformer(nn.Module):
    def __init__(self, num_tokens, k = 128, num_classes = 4, heads = 2, depth = 6, seq_length = 1024):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k, heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(k, num_classes)
        self.do = nn.Dropout(0.1)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                    words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                    classes (where c is the nr. of classes).
        """

        tokens = self.token_emb(x)
        b, t, e = tokens.size()
        
        max_seq_length = max(self.pos_emb.num_embeddings, t)
        positions = torch.arange(max_seq_length, device=x.device)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, e)

        x = tokens + positions
        x = self.tblocks(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
