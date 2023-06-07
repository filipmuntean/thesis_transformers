import torch
import torch.nn as nn
import torch.nn.functional as F
from selfAttention import MultiheadSelfAttention

NUM_TOKENS = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
  
class basicTransformer(nn.Module):
  def __init__(self, num_tokens = 256, k = 128, num_classes = 4, heads = 2, depth = 6,  seq_length = 512):
    super().__init__()

    self.num_tokens = num_tokens
    self.token_embedding = nn.Embedding(num_tokens, k)
    self.pos_embedding = nn.Embedding(seq_length, k)
    self.seq_length = seq_length
    
    tblocks = []
    for i in range(depth):
        tblocks.append(TransformerBlock(k, heads))
    self.tblocks = nn.Sequential(*tblocks)

    # self.toprobs = nn.Linear(k, 16 * 128)
    self.toprobs = nn.Linear(k, num_classes)
    # self.toprobs = nn.Linear(1, 16, 128)
    self.do = nn.Dropout(0.1)

  def forward(self, x):
    """
    :param x: A (batch, sequence length) integer tensor of token indices.
    :return: predicted log-probability vectors for each token based on the preceding tokens.
    """
    tokens = self.token_embedding(x)
    b, t, e = tokens.size()

    positions = self.pos_embedding(torch.arange(t, device=x.device))[None, :, :].expand(b, t, e)
    x = tokens + positions

    x = self.tblocks(x)

    # print(x.shape)
    # x = self.toprobs(x)
    # x = self.toprobs(x.view(b*t, e)).view(b, t, self.num_tokens)
    x = self.toprobs(x.view(b*t, e)).view(b, t, -1)

    # x  = self.toprobs(x.view(b*t, -1)).view(b, t, self.num_tokens)
    # shape '[1, 16, 256]' is invalid for input of size 32768
    return F.log_softmax(x, dim=2)