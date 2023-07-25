import torch
import torch.nn as nn
import torch.nn.functional as F
from selfAttention import MultiheadSelfAttention

NUM_TOKENS = 256

def mask_(matrices, maskval=0.0, mask_diagonal=True):
    """
    Masks out all values in the given batch of matrices where i <= j holds,
    i < j if mask_diagonal is false

    In place operation

    :param tns:
    :return:
    """

    b, h, w = matrices.size()

    indices = torch.triu_indices(h, w, offset=0 if mask_diagonal else 1, dtype=torch.long)
    matrices[:, indices[0], indices[1]] = maskval

class SelfAttentionWide(nn.Module):
    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        self.emb = emb
        self.heads = heads
        self.mask = mask

        self.tokeys = nn.Linear(emb, emb * heads, bias=False)
        self.toqueries = nn.Linear(emb, emb * heads, bias=False)
        self.tovalues = nn.Linear(emb, emb * heads, bias=False)

        self.unifyheads = nn.Linear(heads * emb, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        keys    = self.tokeys(x)   .view(b, t, h, e)
        queries = self.toqueries(x).view(b, t, h, e)
        values  = self.tovalues(x) .view(b, t, h, e)

        # compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, e)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, e)
        values = values.transpose(1, 2).contiguous().view(b * h, t, e)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, e)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, h * e)

        return self.unifyheads(out)
class SelfAttentionNarrow(nn.Module):

    def __init__(self, emb, heads=8, mask=False):
        """

        :param emb:
        :param heads:
        :param mask:
        """

        super().__init__()

        assert emb % heads == 0, f'Embedding dimension ({emb}) should be divisible by nr. of heads ({heads})'

        self.emb = emb
        self.heads = heads
        self.mask = mask

        s = emb // heads
        # - We will break the embedding into `heads` chunks and feed each to a different attention head

        self.tokeys    = nn.Linear(s, s, bias=False)
        self.toqueries = nn.Linear(s, s, bias=False)
        self.tovalues  = nn.Linear(s, s, bias=False)

        self.unifyheads = nn.Linear(heads * s, emb)

    def forward(self, x):

        b, t, e = x.size()
        h = self.heads
        assert e == self.emb, f'Input embedding dim ({e}) should match layer embedding dim ({self.emb})'

        s = e // h
        x = x.view(b, t, h, s)

        keys    = self.tokeys(x)
        queries = self.toqueries(x)
        values  = self.tovalues(x)

        assert keys.size() == (b, t, h, s)
        assert queries.size() == (b, t, h, s)
        assert values.size() == (b, t, h, s)

        # Compute scaled dot-product self-attention

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        queries = queries / (e ** (1/4))
        keys    = keys / (e ** (1/4))
        # - Instead of dividing the dot products by sqrt(e), we scale the keys and values.
        #   This should be more memory efficient

        # - get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))

        assert dot.size() == (b*h, t, t)

        if self.mask: # mask out the upper half of the dot matrix, excluding the diagonal
            mask_(dot, maskval=float('-inf'), mask_diagonal=False)

        dot = F.softmax(dot, dim=2)
        # - dot now has row-wise self-attention probabilities

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, s)

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)
# class alternativeTransformerBlock(nn.Module):
#   def __init__(self, k, heads, mask):
#     super().__init__()
#     self.attention = MultiheadSelfAttention(k = k, heads = heads)
#     self.mask = mask

#     self.norm1 = nn.LayerNorm(k)
#     self.norm2 = nn.LayerNorm(k)

#     self.ff = nn.Sequential(
#       nn.Linear(k, 4 * k),
#       nn.ReLU(),
#       nn.Linear(4 * k, k))

#   def forward(self, x):
#     attended = self.attention(x)
#     x = self.norm1(attended + x)

#     fedforward = self.ff(x)
#     x = self.norm2(fedforward + x)
#     return x
  
# class basicTransformer(nn.Module):
#   def __init__(self, num_tokens = 256, k = 128, num_classes = 4, heads = 2, depth = 6,  seq_length = 512):
#     super().__init__()

#     self.num_tokens = num_tokens
#     self.token_embedding = nn.Embedding(num_tokens, k)
#     self.pos_embedding = nn.Embedding(seq_length, k)
#     self.seq_length = seq_length
    
#     tblocks = []
#     for i in range(depth):
#         tblocks.append(TransformerBlock(k, heads))
#     self.tblocks = nn.Sequential(*tblocks)

#     # self.toprobs = nn.Linear(k, 16 * 128)
#     self.toprobs = nn.Linear(k, num_classes)
#     # self.toprobs = nn.Linear(1, 16, 128)
#     self.do = nn.Dropout(0.1)

#   def forward(self, x):
#     """
#     :param x: A (batch, sequence length) integer tensor of token indices.
#     :return: predicted log-probability vectors for each token based on the preceding tokens.
#     """
#     tokens = self.token_embedding(x)
#     b, t, e = tokens.size()

#     positions = self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)
#     x = tokens + positions

#     x = self.tblocks(x)

#     # print(x.shape)
#     # x = self.toprobs(x)
#     # x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)
#     x = self.toprobs(x.view(b*t, e)).view(b, t, -1)

#     # x  = self.toprobs(x.view(b*t, -1)).view(b, t, self.num_tokens)
#     # shape '[1, 16, 256]' is invalid for input of size 32768
#     return F.log_softmax(x, dim=2)
    
class TransformerBlock(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden=4, dropout=0.0, wide=True):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden * emb, emb)
        )

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        attended = self.attention(x)

        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x