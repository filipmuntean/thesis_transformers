import torch, gzip, os, wget, pickle, random, re, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 32
BLOCK_SIZE = 8
max_iters = 5000
eval_iters = 300
eval_interval = 500
learning_rate = 1e-3
n_embed = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from https://github.com/openai/blocksparse/blob/master/examples/transformer/enwik8.py
    :param path:
    :param n_train:
    :param n_valid:
    :param n_test:
    :return:
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
    :return: the path in which the package resides (the directory containing the 'former' dir)
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def load_data():
    """
    Load the enwik8 dataset from the Hutter challenge.

    Adapted from idk what """
    data = here('/home/mmi349/thesis_transformers/data/enwik8.gz')
    # data = here('filip/thesis/data/enwik8.gz')

    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                            # if final else (data_train, data_val)
    return data_train, data_test

torch.manual_seed(1337)


data_train, data_test = load_data()
# print('data_train:', data_train.shape)
# print('data_test:', data_test.shape)

chars = list(set(data_train.numpy().tolist()))
VOCAB_SIZE = len(chars)
itos = {i: chr(c) for i, c in enumerate(chars)}
decode = lambda l: ''.join([itos[c] for c in l])


def get_batch_vectorized(data, length, batch_size):
    
    # ix = torch.randint(0, data.size(0) - length - 1, (batch_size,))
    ix = torch.randint(len(data) - length, (batch_size,))

    seqs_inputs  = [data[i:i + length] for i in ix]
    
    seqs_target = [data[i + 1:i + length + 1] for i in ix]
    
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    targets = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    
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

        wei = q @ k.transpose(-2, -1) * C**(-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)
    
class FeedForward(nn.Module):
    '''a simple linear layer followed by non-linearity'''

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):

    '''
    A simple bigram language model adapted from:
    Andrej Karpathy -  Let's build GPT: from scratch, with code, spelled out: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2097s
    '''
    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, n_embed)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, n_embed)
        self.sa_heads = MultiHeadAttention(4, n_embed//4)
        self.ff = FeedForward(n_embed)
        self.lm_head = nn.Linear(n_embed, VOCAB_SIZE, bias=False)
    
    def forward(self, idx, targets=None):
        
        B, T = idx.shape

        idx = idx.clamp(max=VOCAB_SIZE-1)
        # print("Min index:", torch.min(idx))
        # print("Max index:", torch.max(idx))

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            targets = targets.clamp(max=VOCAB_SIZE-1)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -BLOCK_SIZE:] # context
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

model = BigramLanguageModel()
# logits, loss = m(xb, yb)
# print(logits.shape, loss)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print("iter {} train loss: {:.2f} test loss: {:.2f}".format(iter, losses[data_train], losses[data_test]))

    xb, yb = get_batch_vectorized(data_train, BLOCK_SIZE, BATCH_SIZE)

    logits, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))

