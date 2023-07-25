import torch, gzip, os, fire, wandb, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import torch.distributions as dist
from transformerModels import basicTransformer, GPT2WrapperRecurrent, GPT2WrapperRegular, GPT2WrapperSimple
# from ytbe import Transformer

# Saved hyperparameters
BATCH_SIZE = 224
BLOCK_SIZE = 8
max_iters = 100000
CONTEXT = 256
TEST_SUBSET = 100000
TEST_BATCH_SIZE = 64
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
LR_WARMUP = 5000
LOGE2 = np.log(2)
# -----------------

# wandb.init(project="Generator on Cluster", entity="filipmuntean", config={"learning_rate": 3e-4, "batch_size": 32})

def sample(lnprobs, temperature=1.0):
    """
    Sample an element from a categorical distribution
    :param lnprobs: Outcome log-probabilities
    :param temperature: Sampling temperature. 1.0 follows the given distribution,
        0.0 returns the maximum probability element.
    :return: The index of the sampled element.
    """

    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)

    return cd.sample()

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
        data_train = torch.from_numpy(train_dataset)
        data_val = torch.from_numpy(val_dataset)
        data_test = torch.from_numpy(test_dataset)
        data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                            if final == True else (data_train, data_val)
        return data_train, data_test if final == True else (data_train, data_val)
    
def here(subpath=None):
    """
    This function was taken and adapted from Peter Bloem -  Transformers from Scratch: https://github.com/pbloem/former/blob/master/experiments/generate.py
    """
    if subpath is None:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

    return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

def alice(path):
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
    
        return str(s[:tr]), str(s[tr:vl]), str(s[vl:])

def load_data():
    """
    Load the enwik8 dataset from the Hutter challenge. This function was taken and adapted from Peter Bloem -  Transformers from Scratch: 
    https://github.com/pbloem/former/blob/master/experiments/generate.py  """

    data = here('/home/mmi349/thesis_transformers/data/enwik8.gz')
    data_train, data_test = enwik8(data) 
    # chars = list(set(data_train.numpy()))
    # VOCAB_SIZE = len(chars)
   
    # # print('VOCAB_SIZE:', VOCAB_SIZE)
    # i2c = {i: chr(c) for i, c in enumerate(chars)}
    # # decode = lambda l: ''.join([i2c[c] for c in l])
    # decode = lambda l: ''.join([i2c.get(c, '') for c in l])
    return data_train, data_test

def load_data_alice():
    data = here('mmi349/thesis_transformers/data/alice.txt')
    data_train, data_val, data_test = alice(data)
    data_train = torch.from_numpy(np.array([ord(c) for c in data_train]))
    data_test = torch.from_numpy(np.array([ord(c) for c in data_test]))
    data_val = torch.from_numpy(np.array([ord(c) for c in data_val]))
        
    data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
                        if final == True else (data_train, data_val)
    return data_train, data_test if final == True else (data_train, data_val)

torch.manual_seed(1337)

data_train, data_test = load_data()

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

def sample_sequence(model, seed, max_context = 256, length=600, temperature=0.5, verbose=True):

    """
    Sequentially samples a sequence from the model, token by token.

    :param model:
    :param seed: The sequence to start with.
    :param length: The total number of characters to sample.
    :param temperature: The sampling temperature.
    :param verbose: If true, the sampled sequence is also printed as it is sampled.
    :return: The sampled sequence, including the seed.
    """

    sequence = seed.detach().clone()

    if verbose: # Print the seed, surrounded by square brackets
        print('[', end='', flush=True)
        for c in seed:
            print(str(chr(c)), end='', flush=True)
        print(']', end='', flush=True)

    for _ in range(length):

        # Input is the tail end of the sampled sequence (as many tokens as the model can handle)
        input = sequence[-max_context:]
        input = input.to(device)
        # Run the current input through the model
        # logits, _ = model(input[None, :])
        logits = model(input[None, :])

        # output_logits = logits[:, -1, :]
        # # Sample the next token from the probabilitys at the last position of the output.
        # print(logits.shape, logits)
        # c = sample(logits[:, -1], temperature)

        c = sample(logits[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence
        # sequence = torch.cat([sequence, c[None]], dim=0)# Append the sampled token to the sequence

    print('\n')
    return seed

@torch.no_grad()
def get_seed(data_test):
    seedfr = random.randint(0, data_test.size(0) - 256)
    print(seedfr);
    seed = data_test[seedfr:seedfr + 256].to(torch.long)
    return seed

@torch.no_grad()
def get_seed_alice(data_test):
    if not isinstance(data_test, torch.Tensor):
        data_test = torch.tensor(data_test)

    # Check if data_test has enough elements
    if data_test.size(0) < 256:
        raise ValueError("data_test should have at least 256 elements")

    seedfr = random.randint(0, data_test.size(0) - 256)
    print(seedfr)
    seed = data_test[seedfr:seedfr + 256].to(torch.long)
    return seed

def d(tensor=None):
    """
    Returns a device string either for the best available device,
    or for the device corresponding to the argument
    :param tensor:
    :return:
    """
    if tensor is None:
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return 'cuda' if tensor.is_cuda else 'cpu'

def compute_compression(model, data, context, batch_size, verbose=False,
                        skip=0):


    """
    Compute the _compression_ of a dataset under a model. That is, given a model, in how many bits could we represent
    the dataset. This requires us to turn a given probability distribution into a code for the outcomes.

    See [this video](https://youtu.be/mSneVjDvzNQ) for an explanation.

    :param model: A sequence-to-sequence model that takes as input a (sub) sequence of integers and produces a probability
    distributuion on the output.
    :param data: A singe list of integers representing the  data
    :return: The result of the computation in "bits per byte". That is, how many bits does the compressed representation
    spend on each byte (=ASCII character) of the raw data.
    """

    bits, tot = 0.0, 0
    batch = []
    # Buffer, every time it fills up, we run it through the model
    # --- For the sake of speed we want to process the data in batches. For each token in the data, we make a
    #     prediction based on all the `context` tokens before it. This means that for each subsequence in the batch, we
    #     need to shift the start/end indices ahead by one token.
    #
    #     After we pass the batch through the model, we look at only the probabilities predicted for the last token.

    target_indices = []
    i, ic = 0, 0

    for current in range(skip, data.size(0)) if verbose else range(skip, data.size(0)):

        # `current` is the character which we will ultimately predict

        fr = max(0, current - context)
        to = current + 1

        instance = data[fr:to].to(torch.long) # the subsequence of the data to add to the batch
        # -- slice out an instance of size context + 1 (or shorter at the start of the data)

        # if tok is not None:
        #     print(instance[:-1], tok.decode(instance[:-1]))
        #     print(instance[-1:], tok.decode(instance[-1:]))

        target_indices.append(instance.size(0) - 2) # index of the last element of the input to the model

        if instance.size(0) < context + 1:
            assert skip < context # We shouldn't get here if we skip the first `context` characters

            # the index in the output tensor of the character we want to predict
            # -- It's context + 1, because we clip off the last token as a target

            pad = torch.zeros(size=(context + 1 - instance.size(0),), dtype=torch.long)
            instance = torch.cat([instance, pad], dim=0)
            # -- the first tokens don't have enough tokens preceding them, so we pad them to the right size.

            assert instance.size(0) == context + 1 # all instances should be `context` + 1 long

        if torch.cuda.is_available():
            instance = instance.cuda()

        batch.append(instance[None, :])
        # -- We add a singleton dimension to concatenate along later.

        if len(batch) == batch_size or current == data.size(0) - 1:
            # batch is full or we are at the last instance, run it through the model

            b = len(batch)

            ti = torch.tensor(target_indices) + 1
            all = torch.cat(batch, dim=0)
            inputs = all[:, :-1] # input
            target = all[torch.arange(b), ti]  # target values

            with torch.no_grad():
                if next(model.parameters()).is_cuda:
                    inputs = inputs.cuda()
                output = model(inputs)

            if type(output) != torch.Tensor:
                output = torch.log_softmax(output.logits, dim=2) # To make the method work for GPT2 models from Huggingface

            assert output.size()[:2] == (b, context), f'was: {output.size()}, should be {(b, context, -1)}'

            lnprobs = output[torch.arange(b, device=d()), target_indices, target]
            log2probs = lnprobs / LOGE2
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilities) to the running total
            batch, target_indices = [], []  # clear the buffer

    if isinstance(bits, torch.Tensor):
        bits = bits.item()

    return bits # total nr of bits used

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in [data_train, data_test]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            source, target = get_batch_vectorized(split, BLOCK_SIZE, BATCH_SIZE)
            
            output = model(source)
            
            # Resize the target tensor to match the output tensor's shape
            target = target[:, :output.size(1)]  # Trim target to match output length
            
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
            # loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model = GPT2WrapperRecurrent(iblocks = 3, gptname='distilgpt2')
def go(model, optimizer, sch):
    for iter in range(max_iters):

        if iter % eval_interval == 0:
            seed = get_seed(data_test)
            seed = seed.cuda()

            sample_sequence(model, seed = seed, verbose=True, max_context=768, length=1000)
            losses = estimate_loss(model)

            upto = data_test.size(0) if iter == max_iters - 1 else TEST_SUBSET
            data_sub = data_test[:upto]
            bits_per_byte = compute_compression(model, data_sub, context=CONTEXT, batch_size=TEST_BATCH_SIZE)
            
            print("iter {} train loss: {:.2f} test loss: {:.2f}".format(iter, losses[data_train], losses[data_test]))
            print("iter {} test bits per byte: {:.2f}".format(iter, bits_per_byte))
            # wandb.log({"iter": iter, "WITH RECURRENT CONNECTION: train loss batch size 256, block size 1, 1000 iters, eval_interval 500": losses[data_train], "WITH RECURRENT CONNECTION: test loss batch size 256, block size 1, 1000 iters, eval_interval 500": losses[data_test]})

        source, target = get_batch_vectorized(data_train, BLOCK_SIZE, BATCH_SIZE)
        # logits, loss = model(xb, yb)
        output = model(source)
        # loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        loss = F.cross_entropy(output.transpose(2,1), target, reduction='mean')

        # Set all gradients of the model parameters to zero.
        optimizer.zero_grad(set_to_none=True)
        # set_to_none = True is a memory-efficient way to zero out the gradients
        loss.backward()
        grad_clip = clip_grad_norm_(model.parameters(), max_norm=1.0)
        # wandb.log({"gradient clipping": grad_clip})
        optimizer.step()
        # sch.step()

# context = torch.zeros((1, 1), dtype=torch.long, device = device)
# generated_txt = (model.generate(context, max_new_tokens=2000)[0].tolist())
# decoded_txt = decode(generated_txt)

# print(decoded_txt)

# wandb.log({"generated text": generated_txt})
# open('more.txt', 'w').write(decode(model.generate(torch.zeros((1, 1), dtype=torch.long, device = device), max_new_tokens=10000)[0].tolist()))

# wandb.finish()

class Handler(object):

    def __init__(self, generator="instances"):
        super().__init__()
        self.generator = generator

    @staticmethod
    def run(generator="basic"):
        if generator == "basic":
            print("=====================BASIC TRANSFORMER=====================")
            model = basicTransformer()
            model = model.to(device)
            # wandb.watch(model)
            print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters' + "\n")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))
            go(model, optimizer, sch)
        elif generator == "pretrained":
            print("===================PRETRINED TRANSFORMER=====================")
            model = GPT2WrapperRegular(iblocks = 3, gptname='distilgpt2')
            model = model.to(device)
            # wandb.watch(model)
            print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters' + "\n")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))
            go(model, optimizer, sch)
        elif generator == "recurrent":
            print("===================RECURRENT TRANSFORMER=====================")
            model = GPT2WrapperRecurrent(iblocks = 3, gptname='distilgpt2')
            model = model.to(device)
            # wandb.watch(model)
            print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters' + "\n")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))
            go(model, optimizer, sch)
        elif generator == "simple":
            print("===================SIMPLE TRANSFORMER=====================")
            model = GPT2WrapperSimple(iblocks = 3, gptname='distilgpt2')
            model = model.to(device)
            # wandb.watch(model)
            print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters' + "\n")
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            sch = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (LR_WARMUP / BATCH_SIZE), 1.0))
            go(model, optimizer, sch)
        else:
            print("Should be max or mean pooling")
            return 1
    
        
if __name__ == '__main__':
    # check for args
    import sys
    if len(sys.argv) > 1:
        fire.Fire(Handler)
    else:
        Handler.run("simple")
