import torch, gzip, os, wget, pickle, random, re, sys
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

ascii_art = """

Let's generate some text!

              .:.               
             .::::.             
..         ..::::::''::         
::::..  .::''''':::    ''.      
':::::::'         '.  ..  '.    
 ::::::'            : '::   :   
  :::::     .        : ':'   :  
  :::::    :::       :.     .' 
 .::::::    ':'     .' '.:::: : 
 ::::::::.         .    ::::: : 
:::::    '':.... ''      '''' : 
':::: .:'              ...'' :  
 ..::.   '.........:::::'   :   
  '':::.   '::'':'':::'   .'    
        '..  ''.....'  ..'      
           ''........''
            """

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
    data = here('filip/thesis/data/enwik8.gz')

    data_train, data_val, data_test = enwik8(data)
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                            # if final else (data_train, data_val)
    return data_train, data_test

torch.manual_seed(1337)

# print(ascii_art, "\n")

data_train, data_test = load_data()
# print('data_train:', data_train.shape)
# print('data_test:', data_test.shape)

chars = list(set(data_train.numpy()))
vocab_size = len(chars)
itos = {i: chr(c) for i, c in enumerate(chars)}
decode = lambda l: ''.join([itos[c] for c in l])
BATCH_SIZE = 4
BLOCK_SIZE = 8

def get_batch_vectorized(data, length, batch_size):
    
    ix = torch.randint(data.size(0) - length, (batch_size,))

    seqs_inputs  = [data[i:i + length] for i in ix]
    
    seqs_target = [data[i + 1:i + length + 1] for i in ix]
    
    inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    targets = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    
    return inputs, targets

xb, yb = get_batch_vectorized(data_train, BLOCK_SIZE, BATCH_SIZE)
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

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):
        
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits, loss = self(idx)
            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=-1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
print(logits.shape, loss)


print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))