import torch
import torch.nn as nn
import torch.optim as optim
from main import Main
from tokens import Tokens
import itertools

class AlternativeSelfAttention(nn.Module):
    def __init__(self, embed_size, heads = 4, mask = False):
        super(AlternativeSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * heads == embed_size), "Embed size needs to be divisible by heads"
       
        self.tokeys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.toqueries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.tovalues = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.unifyheads = nn.Linear(heads * self.head_dim, embed_size)

        self.mask = mask

    def forward(self, values, keys, query, mask):
        H = self.heads
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, H, self.head_dim)
        keys = keys.reshape(N, key_len, H, self.head_dim)
        query = query.reshape(N, query_len, H, self.head_dim)

        keys = self.tokeys(keys)
        query = self.toqueries(query)
        values = self.tovalues(values)

        energy = torch.einsum("nqhd,nkhd->nhqk", [query, keys])

        if self.mask is not None:
            energy = energy.masked_fill(self.mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim)
        out = self.unifyheads(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = AlternativeSelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout = dropout,
                    forward_expansion = forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        print(x.shape, self.word_embedding(x).shape, self.position_embedding(positions).shape, out.shape)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        
        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = AlternativeSelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, 
                                                  heads, 
                                                  dropout, 
                                                  forward_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out  
    
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, 
                embed_size, num_layers, heads,
                forward_expansion, dropout, device, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)])

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        
        out = self.fc_out(x)
        return out
    
class transformer(nn.Module):
    def __init__(self, src_vocab_size, 
                trg_vocab_size, 
                src_pad_idx, 
                trg_pad_idx, 
                embed_size = 256, 
                num_layers = 6, 
                forward_expansion = 4, 
                heads = 8, 
                dropout = 0, 
                device = 'meta', 
                max_length = 100):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, 
                               embed_size, 
                               num_layers, 
                               heads, 
                               device, 
                               forward_expansion, 
                               dropout, 
                               max_length)
        self.decoder = Decoder(trg_vocab_size,
                                embed_size,
                                num_layers,
                                heads,
                                forward_expansion,
                                dropout,
                                device,
                                max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_length = trg.shape
        trg_mask = torch.tril(torch.ones((trg_length, trg_length))).expand(N, 1, 
                                                                           trg_length, 
                                                                           trg_length)
        return trg_mask.to(self.device)
    
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

