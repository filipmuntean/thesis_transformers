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
  
class basictransformer(nn.Module):
    def __init__(self, num_tokens, k = 128, num_classes = 4, heads = 2, depth = 6, seq_length = 512):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)
        self.seq_length = seq_length
        
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
    
        positions = torch.arange(t)
        positions = positions.reshape(-1)
        positions = self.pos_emb(positions).unsqueeze(0).expand(b, t, e)
        # positions = self.pos_emb(positions)[None, :, :].expand(b, t, e)
        x = tokens + positions
        x = self.tblocks(x)

        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)