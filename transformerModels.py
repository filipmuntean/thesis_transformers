import torch
import torch.nn as nn
import torch.nn.functional as F
from selfAttention import SelfAttentionNarrow, SelfAttentionWide
from transformers import GPT2Tokenizer, GPT2LMHeadModel

NUM_TOKENS = 256
device = torch.device('cuda')
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
  
class basicTransformer(nn.Module):
  def __init__(self, num_tokens = 256, k = 128, num_classes = 4, heads = 2, depth = 6,  seq_length = 512, mask = None):
    super().__init__()

    self.num_tokens = num_tokens
    self.token_embedding = nn.Embedding(num_tokens, k)
    self.pos_embedding = nn.Embedding(seq_length, k)
    self.seq_length = seq_length
    
    tblocks = []
    for i in range(depth):
        tblocks.append(TransformerBlock(k, heads, mask))
    self.tblocks = nn.Sequential(*tblocks)

    # self.toprobs = nn.Linear(k, 16 * 128)
    self.toprobs = nn.Linear(k, num_classes)
    # self.toprobs = nn.Linear(1, 16, 128)
    self.do = nn.Dropout(0.1)

  def forward(self, x, mask = None):
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

        # attended = self.attention(x)
        attended = self.attention(x.to(device)).to(device)
        x = x.to(device)
        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        return x
    
class TransformerBlockRecurrent(nn.Module):

    def __init__(self, emb, heads, mask, ff_hidden_mult=4, dropout=0.0, wide=True, encoder_hidden_states = None):
        super().__init__()

        self.attention = SelfAttentionWide(emb, heads=heads, mask=mask) if wide \
                    else SelfAttentionNarrow(emb, heads=heads, mask=mask)
        self.mask = mask

        self.norm1 = nn.LayerNorm(emb)
        self.norm2 = nn.LayerNorm(emb)

        self.ff = nn.Sequential(
            nn.Linear(emb, ff_hidden_mult * emb),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb, emb)
        )
        self.recurrent = nn.GRU(emb, emb, batch_first=True)

        self.do = nn.Dropout(dropout)

    def forward(self, x, recurrent_input=None):

        # attended = self.attention(x)
        attended = self.attention(x.to(device)).to(device)
        x = x.to(device)
        x = self.norm1(attended + x)

        x = self.do(x)

        fedforward = self.ff(x)

        x = self.norm2(fedforward + x)

        x = self.do(x)

        if recurrent_input is not None:
            recurrent_output, _ = self.recurrent(recurrent_input.to(device))
            x += recurrent_output

        return x
    
class NoParam(nn.Module):
    """
    Wraps a module, stopping parameters from being registered
    """

    def __init__(self, mod):
        super().__init__()
        self.mod = [mod]

    def cuda(self):
        self.mod[0].cuda()

    def forward(self, x, *args, **kwargs):
        new_kwargs = {k: v for k, v in kwargs.items() if k not in ["encoder_hidden_states"]}
        output = self.mod[0](x, *args, **new_kwargs)
        # output = output.to(x.device)
        return output

class IBlock(nn.Module):
    """
    Transformer block to be inserted into GPT2 stack. Allows conditionals
    to be registered prior to forward.
    """

    def __init__(self, emb, *args, mult=0.0, csize=None, cond=[None], **kwargs):

        super().__init__()
        self.block = TransformerBlock(emb, *args, **kwargs)
        self.mult = nn.Parameter(torch.tensor([mult]))

        self.cond = cond
        self.cond_out = [None]

        if csize is not None:
            self.to_cond = nn.Sequential(
                nn.Linear(csize, 2 * csize), nn.ReLU(),
                nn.Linear(2 * csize, emb)
            )

        # self.to_cond = nn.Linear(csize, emb)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None):

        b, l, e = x.size()

        if self.cond is not None and len(self.cond) > 0 and self.cond[0] is not None:
            cond = self.to_cond(self.cond[0])
            assert cond.size() == (b, e), f'{cond.size()} versus {b, e}'

            self.cond_out[0] = cond

            xc = x + cond[:, None, :]
        else:
            xc = x

        r = self.mult * self.block(xc) +  x

        # print(r.size())
        return r, None, None

    def clear(self):
        del self.cond_out[0]
        del self.cond_out
        self.cond_out = [None]

        # del self.cond[0]
        # del self.cond
        # self.cond = [None]

class IBlockRecurrent(nn.Module):
    """
    Transformer block to be inserted into GPT2 stack. Allows conditionals
    to be registered prior to forward.
    """

    def __init__(self, emb, *args, mult=0.0, csize=None, cond=[None], **kwargs):

        super().__init__()
        self.block = TransformerBlockRecurrent(emb, *args, **kwargs)
        self.mult = nn.Parameter(torch.tensor([mult]))

        self.cond = cond
        self.cond_out = [None]

        if csize is not None:
            self.to_cond = nn.Sequential(
                nn.Linear(csize, 2 * csize), nn.ReLU(),
                nn.Linear(2 * csize, emb)
            )
            self.to_cond.to(device) 
            # self.to_cond = nn.Linear(csize, emb)

    def forward(self, x, recurrent_input=None, layer_past=None, attention_mask=None, head_mask=None):

        b, l, e = x.size()

        if self.cond is not None and len(self.cond) > 0 and self.cond[0] is not None:
            if hasattr(self, 'to_cond'):
                cond = self.to_cond(self.cond[0].to(device))
                assert cond.size() == (b, e), f'{cond.size()} versus {b, e}'
            else:
                raise AttributeError("Conditional size is not defined.")

            self.cond_out[0] = cond

            xc = x + cond[:, None, :]
        else:
            xc = x

        if recurrent_input is not None:
            xc = xc.to(device)
            recurrent_input = recurrent_input.to(device)

        r = self.mult * self.block(xc, recurrent_input) + x.to(device)

        # r = self.mult * self.block(xc) +  x

        # print(r.size())
        return r, None, None

    def clear(self):
        del self.cond_out[0]
        del self.cond_out
        self.cond_out = [None]

        # del self.cond[0]
        # del self.cond
        # self.cond = [None]

class GPT2WrapperRecurrent(nn.Module):

    def __init__(self, iblocks=3, gptname='distilgpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname)
        model = GPT2LMHeadModel.from_pretrained(gptname, output_hidden_states=True)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        emb = model.config.n_embd
        self.ctx = model.config.n_ctx

        self.container = [None]

        self.iblocks = nn.ModuleList([
            IBlockRecurrent(emb=emb, heads=8, mask=True, ff_hidden_mult= 4, dropout=dropout, wide=False, csize=csize, cond=self.container) for _ in range(iblocks+1)
        ])

        nb = len(model.transformer.h)   # number of GPT2 blocks
        per = nb // iblocks             # how many blocks to skip between inserted blocks

        h = model.transformer.h # the main stack of transformer blocks
        for i in range(iblocks-1, -1, -1):
            print('inserting block at', i*per)
            block = self.iblocks[i]
            h.insert(i*per, block)
        h.insert(len(h), self.iblocks[-1])

        self.register_buffer(name='head_mask', tensor=torch.ones(len(h), model.config.n_head))
        # We need to create a special head mask because we've changes the number of blocks.

        self.model = NoParam(model)

        # Out own language model head
        self.headbias = nn.Parameter(torch.zeros(self.tokenizer.vocab_size)) # to token probabilities

        # if csize is not None:
        #     self.to_cond = nn.Sequential(
        #         nn.Linear(csize, 2*csize), nn.ReLU(),
        #         nn.Linear(2*csize, emb)
        #     )
    def forward(self, x, cond=None):
        
        b = x.size(0)

        if cond is not None:
            self.container[0] = cond

        x = x.to(next(self.model.mod[0].parameters()).device)
        x = self.model(x, head_mask=self.head_mask)[0]
        x = x.to(x.device)
        return x + self.headbias
        # x =  0.0 * cond.view(b, -1).sum(dim=1) #hack


    def clear(self):

        del self.container[0]
        del self.container
        self.container = [None]

        for block in self.iblocks:
            block.clear()

class GPT2WrapperRegular(nn.Module):

    def __init__(self, iblocks=3, gptname='distilgpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname)
        model = GPT2LMHeadModel.from_pretrained(gptname)
        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        emb = model.config.n_embd
        self.ctx = model.config.n_ctx

        self.container = [None]

        self.iblocks = nn.ModuleList([
            IBlock(emb=emb, heads=8, mask=True, ff_hidden=4, dropout=dropout, wide=False, csize=csize, cond=self.container) for _ in range(iblocks+1)
        ])

        nb = len(model.transformer.h)   # number of GPT2 blocks
        per = nb // iblocks             # how many blocks to skip between inserted blocks

        h = model.transformer.h # the main stack of transformer blocks
        for i in range(iblocks-1, -1, -1):
            print('inserting block at', i*per)
            block = self.iblocks[i]
            h.insert(i*per, block)
        h.insert(len(h), self.iblocks[-1])

        self.register_buffer(name='head_mask', tensor=torch.ones(len(h), model.config.n_head))
        # We need to create a special head mask because we've changes the number of blocks.

        self.model = NoParam(model)

        # Out own language model head
        self.headbias = nn.Parameter(torch.zeros(self.tokenizer.vocab_size)) # to token probabilities

        # if csize is not None:
        #     self.to_cond = nn.Sequential(
        #         nn.Linear(csize, 2*csize), nn.ReLU(),
        #         nn.Linear(2*csize, emb)
        #     )

    def forward(self, x, cond=None):

        b = x.size(0)

        if cond is not None:
            self.container[0] = cond

        x = self.model(x, head_mask=self.head_mask)[0]
        # x =  0.0 * cond.view(b, -1).sum(dim=1) #hack

        return x + self.headbias

    def clear(self):

        del self.container[0]
        del self.container
        self.container = [None]

        for block in self.iblocks:
            block.clear()

class GPT2WrapperSimple(nn.Module):
    def __init__(self, iblocks=3, gptname='distilgpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname)
        self.model = GPT2LMHeadModel.from_pretrained(gptname)
        self.model.eval()
        # self.iblocks = nn.ModuleList(list(self.model.iblocks()))

        self.emb = self.model.config.n_embd
        self.ctx = self.model.config.n_ctx
        
        self.recurrent = nn.GRU(68*self.emb - 1967, self.emb, num_layers=1, batch_first=True)
        self.lm_head = nn.Linear(self.emb, self.model.config.vocab_size, bias=True)

    def forward(self, x, cond=None):
        x = self.model(x, cond)[0]
        x = x[:, :self.ctx, :]  # Trim to max context length

        # # if cond is not None:
        # batch_size, seq_len, _ = x.size()
        # cond = cond.unsqueeze(1).expand(-1, seq_len, -1)  # Expand cond tensor to match the sequence length
        # x = torch.cat((x, cond), dim=-1)  # Concatenate x and cond along the last dimension

        # # Reshape the input tensor for the GRU
        # batch_size, seq_len, feature_size = x.size()
        # x = x.view(batch_size * seq_len, feature_size)

        x, _ = self.recurrent(x)  # Apply the recurrent connection using GRU

        # Reshape the output tensor back to the original shape
        # x = x.view(batch_size, seq_len, feature_size)
        x = x[:, -1, :]
        x = F.relu(x)
        x = self.lm_head(x)

        return x