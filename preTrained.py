from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import os , gzip, torch
import numpy as np
import wandb, random, gc
from torch import nn
from basicTransformer import TransformerBlock


wandb.init(project="preTrained Generator on Cluster", entity="filipmuntean", config={"learning_rate": 3e-4, "batch_size": 32})

# Hyperparameters
model_name = 'gpt2'
final = True
device = torch.device('cuda')
torch.cuda.empty_cache()
gc.collect()

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

        return self.mod[0](x, *args, **kwargs)
    

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

class GPT2Wrapper(nn.Module):

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
            IBlock(emb=emb, heads=8, mask=True, ff_hidden_mult=4, dropout=dropout, wide=False, csize=csize, cond=self.container) for _ in range(iblocks+1)
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

def load_text(path):
    """
    Load text data.

    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
    """
    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        s = file.read()
        tr, vl = int(len(s) * 0.6), int(len(s) * 0.8)

        return str(s[:tr]), str(s[tr:vl]), str(s[vl:])\
        
