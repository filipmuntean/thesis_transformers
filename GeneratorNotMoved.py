import torch, gzip, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import wandb, random
# hyperparameters
BATCH_SIZE = 224
BLOCK_SIZE = 128
max_iters = 2000
eval_interval = 500
eval_iters = 200
learning_rate = 3e-4
n_embed = 384
n_layer = 12
n_head = 8
dropout = 0.2
weight_decay = 1e-5
device = torch.device('cuda') 
final = True
VOCAB_SIZE = 256
# -----------------

wandb.init(project="Generator on Cluster", entity="filipmuntean", config={"learning_rate": 3e-4, "batch_size": 32})

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    This function was taken and adapted from Peter Bloem - Transformers from Scratch: https://github.com/pbloem/former/blob/master/experiments/generate.py
    """

    if path is None:
        path = here('data/enwik8.gz')

    with gzip.open(path) if path.endswith('.gz') else open(path) as file:
        X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
        trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
        train_dataset = np.copy(trX)
        val_dataset = np.copy(vaX)
        test_dataset = np.copy(teX)
        return torch.from_numpy(train_dataset), torch.from_numpy(val_dataset), torch.from_numpy(test_dataset)
    
def here(subpath=None):
    """
    This function was taken and adapted from Peter Bloem -  Transformers from Scratch: https://github.com/pbloem/former/blob/master/experiments/generate.py
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def load_data():
    """
    Load the enwik8 dataset from the Hutter challenge. This function was taken and adapted from Peter Bloem -  Transformers from Scratch: 
    https://github.com/pbloem/former/blob/master/experiments/generate.py  """

    data = here('/home/mmi349/thesis_transformers/data/enwik8.gz')
    # data = here('filip/thesis/data/enwik8.gz')

    data_train, data_val, data_test = enwik8(data) 
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                            if final == True else (data_train, data_val)

    chars = list(set(data_train.numpy()))
    VOCAB_SIZE = len(chars)
   
    # print('VOCAB_SIZE:', VOCAB_SIZE)
    i2c = {i: chr(c) for i, c in enumerate(chars)}
    decode = lambda l: ''.join([i2c[c] for c in l])
    return data_train, data_test, chars, i2c, decode

torch.manual_seed(1337)

data_train, data_test, chars, i2c, decode = load_data()
def get_batch_vectorized(data, length, batch_size):

    '''This function was taken and adapted from Peter Bloem -  Transformers from Scratch: 
    https://github.com/pbloem/former/blob/master/experiments/generate.py'''
    
    ix = torch.randint(0, data.size(0) - length - 1, (batch_size,))

    seqs_inputs  = [data[i:i + length] for i in ix]
    seqs_target = [data[i + 1:i + length + 1] for i in ix]
    
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    targets = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in [data_train, data_test]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch_vectorized(split, BLOCK_SIZE, BATCH_SIZE)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# print('inputs:')
# print(xb.shape, xb)
# print('\ntargets:')
# print(yb.shape, yb)
# print('-------')
# for b in range(BATCH_SIZE):
#     for t in range(BLOCK_SIZE):
#         context = xb[b, :t + 1]
#         target = yb[b, t]
#         print(f"when input is {context} the target is {target}")

class Head(nn.Module):
    ''' one head self attention'''

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        wei = (q @ k.transpose(-2, -1)) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
    
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    '''a simple linear layer followed by non-linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    ''' Transformer block '''

    def __init__(self, n_embed, n_head):
        super().__init__()

        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Make sure to add residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        x = self.dropout(x)
        return x

class Transformer(nn.Module):

    '''
    A simple transformer model adapted from:
    Andrej Karpathy -  Let's build GPT: from scratch, with code, spelled out: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2097s
    '''
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, n_embed)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head = n_head) for _ in range(n_layer)],
        )
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, VOCAB_SIZE, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx = idx.clamp(max=VOCAB_SIZE - 1)
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # targets = targets.clamp(max=VOCAB_SIZE-1)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = Transformer()
model = model.to(device)
# wandb.watch(model)
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print("iter {} train loss: {:.2f} test loss: {:.2f}".format(iter, losses[data_train], losses[data_test]))
        wandb.log({"iter": iter, "train loss batch size 224, block size 128, 2000 iters, eval_interval 500": losses[data_train], "test loss batch size 224, block size 128, 2000 iters, eval_interval 500": losses[data_test]})

    xb, yb = get_batch_vectorized(data_train, BLOCK_SIZE, BATCH_SIZE)
    logits, loss = model(xb, yb)
    # Set all gradients of the model parameters to zero.
    optimizer.zero_grad(set_to_none=True)
    # set_to_none = True is a memory-efficient way to zero out the gradients
    loss.backward()
    grad_clip = clip_grad_norm_(model.parameters(), max_norm=1.0)
    wandb.log({"gradient clipping": grad_clip})
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device = device)
generated_txt = print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

wandb.log({"generated text": generated_txt})
# open('more.txt', 'w').write(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device = device), max_new_tokens=10000)[0].tolist()))

wandb.finish()
