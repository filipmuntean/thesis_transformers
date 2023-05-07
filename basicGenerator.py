import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import random, tqdm, sys, math
from argparse import ArgumentParser
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from basicTransformer import transformer
from main import Main

NUM_TOKENS = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def sample_batch(data, length, batch_size):
    """
    Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

    For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
    target for the model.

    :param data: The (training) data. A single vector of tokens represented by integers
    :param length: The length of the subsequences in the batch.
    :param batch_size: The number of subsequences in the batch
    :return: A pair (input, target) of minteger matrices representing the input and target for the model.
    """

    # starts = torch.randint(size=(batch_size,), low=0, high=data.size(0) - length - 1)

    # # Slice out the input sequences
    # seqs_inputs  = [data[start:start + length] for start in starts]
    # # -- the start index is the one we just sampled, and the end is exactly 'lentgh' positions after that.
    # seqs_target = [data[start + 1:start + length + 1] for start in starts]
    # # -- The target is the same sequence as input, except one character ahead (we are asking the model to predict the
    # #    next character at each position)

    # # We now have two lists of torch vectors, which we can concatenate into matrices of batch_size-by-length
    # inputs = torch.cat([s[None, :] for s in seqs_inputs], dim=0).to(torch.long)
    # target = torch.cat([s[None, :] for s in seqs_target], dim=0).to(torch.long)
    # # -- Note that we add a singleton dimenson to each vector, s[None.,:], and then concatenate along that dimension.

    # return inputs, target

    max_size = max(t.size(1) for t in data)
    padded_tensors = []
    for t in data:
        pad = (0, max_size - t.size(1))
        padded_tensors.append(torch.nn.functional.pad(t, pad, mode='constant'))

    # stack the padded tensors along the batch dimension
    padded_inputs = torch.stack(padded_tensors[start:start + length] for start in padded_tensors)
    padded_targets = torch.stack(padded_tensors[start + 1:start + length + 1] for start in padded_tensors)

    inputs = torch.cat([s[None, :] for s in padded_inputs], dim=0).to(torch.long)
    targets = torch.cat([s[None, :] for s in padded_targets], dim=0).to(torch.long)
    
    return inputs, targets

def sample_sequence(model, seed, max_context, length=600, temperature=0.5, verbose=False):
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

        # Run the current input through the model
        output = model(input[None, :])

        # Sample the next token from the probabilitys at the last position of the output.
        c = sample(output[0, -1, :], temperature)

        if verbose:
            print(str(chr(max(32, c))), end='', flush=True)

        sequence = torch.cat([sequence, c[None]], dim=0) # Append the sampled token to the sequence

    print()
    return seed

def compute_compression(model, data, context, batch_size, verbose=False,
                    tbw:SummaryWriter=None, tok=None, skip=0):
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

    for current in tqdm.trange(skip, data.size(0)) if verbose else range(skip, data.size(0)):

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

            lnprobs = output[torch.arange(b, device='cuda'), target_indices, target]
            log2probs = lnprobs / math.log(2.0)
            # -- The model produces natural logarithms of probabilities, but we need base-2 logarithms of the
            #    probabilities, since these give us bits.

            if tbw is not None:
                for j, lp in enumerate(log2probs):
                    i += 1
                    tbw.add_scalar('compression/bits-per-token', -lp, i)

                    if tok is not None:
                        nc = len(tok.decode(target[j]))
                        ic += nc
                        tbw.add_scalar('compression/bits-per-byte', -lp/nc, ic)

            bits += - log2probs.sum() # Add the bits for each character (the negative log_2 probabilities) to the running total
            batch, target_indices = [], []  # clear the buffer

    if isinstance(bits, torch.Tensor):
        bits = bits.item()

    return bits # total nr of bits used

def go(arg):

    if arg.seed < 0:
        seed = random.randint(0, 1000)
        print('random seed: ', seed)
    else:
        torch.manual_seed(arg.seed)

    # tbw = SummaryWriter(log_dir=arg.tb_dir) # Tensorboard logging

    # arg.data = here('data/enwik8.gz') if arg.data is None else arg.data

    data_train = Main.x_train
    data_train, data_test = (torch.cat([Main.x_train, Main.x_val], dim=0), data_test) \
                            if arg.final else (data_train, Main.x_val)

    # create the model
    model = transformer(k=arg.embedding_size, heads=arg.num_heads, depth=arg.depth, seq_length=arg.context, num_tokens=NUM_TOKENS)
    if torch.cuda.is_available():
        model.cuda()

    opt = torch.optim.Adam(lr=arg.lr, params=model.parameters())

    # Linear learning rate warmup
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))

    # Training loop
    # -- We don't loop over the data, instead we sample a batch of random subsequences each time. This is not strictly
    #    better or worse as a training method, it's just a little simpler.
    
    instances_seen = 0
    for i in tqdm.trange(arg.num_batches):

        opt.zero_grad()
        
        source, target = sample_batch(Main.review_tensors, length=arg.context, batch_size=16)
        instances_seen += source.size(0)

        if torch.cuda.is_available():
            source, target = source.cuda(), target.cuda()

        # tic()
        output = model(source) # forward pass
        # t = toc()

        # Compute the loss
        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')

        # tbw.add_scalar('transformer/train-loss', float(loss.item()) * math.log(2.0), i * arg.batch_size, instances_seen)
        # tbw.add_scalar('transformer/time-forward', t, instances_seen)

        loss.backward() # backward pass

        # clip gradients
        # -- If the total gradient vector has a length > x, we clip it back down to x.
        if arg.gradient_clipping > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), arg.gradient_clipping)

        opt.step() # stochastic gradient descent step
        sch.step() # update the learning rate

        # Validate every `arg.test_every` steps. First we compute the
        # compression on the validation data (or a subset),
        # then we generate some random text to monitor progress.
        if i != 0 and (i % arg.test_every == 0 or i == arg.num_batches - 1):
            with torch.no_grad():

                ## Sample and print a random sequence

                # Slice a random seed from the test data, and sample a continuation from the model.
                seedfr = random.randint(0, Main.test_dataset.size(0) - arg.context)
                seed = Main.test_dataset[seedfr:seedfr + arg.context].to(torch.long)

                if torch.cuda.is_available():
                    seed = seed.cuda()

                sample_sequence(model, seed=seed, max_context=arg.context, verbose=True, length=arg.sample_length)

                ## Compute validation bits per byte

                upto = Main.test_dataset.size(0) if i == arg.num_batches - 1 else arg.test_subset
                data_sub = Main.test[:upto]

                bits_per_byte = compute_compression(model, data_sub, context=arg.context, batch_size=arg.test_batchsize)
                # -- Since we're not computing gradients, we can increase the batch size a little from what we used in
                #    training.

                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
                # tbw.add_scalar(f'transformer/eval-loss', bits_per_byte, i * arg.batch_size, instances_seen)
                # -- 0.9 bit per byte is around the state of the art.

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-N", "--num-batches",
                        dest="num_batches",
                        help="Number of batches to train on. Each batch contains randomly sampled subsequences of the data."
                             "Default is set to a very large value so you can keep running until the output looks good. ",
                        default=3, type=int)

    parser.add_argument("-b", "--batch-size",
                        dest="batch_size",
                        help="The batch size.",
                        default=32, type=int)

    parser.add_argument("-D", "--data", dest="data",
                        help="Data file. Will be read as a string of 8-bit characters.",
                        default=None)

    parser.add_argument("-l", "--learn-rate",
                        dest="lr",
                        help="Learning rate",
                        default=0.0001, type=float)

    parser.add_argument("-T", "--tb-dir", dest="tb_dir",
                        help="Tensorboard logging directory",
                        default='./runs')

    parser.add_argument("-f", "--final", dest="final",
                        help="Whether to run on the real test set (if not included, the validation set is used).",
                        action="store_true")

    parser.add_argument("-E", "--embedding", dest="embedding_size",
                        help="Size of the character embeddings.",
                        default=128, type=int)

    parser.add_argument("-H", "--heads", dest="num_heads",
                        help="Number of attention heads.",
                        default=8, type=int)

    parser.add_argument("-C", "--context", dest="context",
                        help="Length of the sequences extracted from the corpus (and the context used during inference).",
                        default=256, type=int)

    parser.add_argument("-d", "--depth", dest="depth",
                        help="Depth of the network (nr. of transformer blocks)",
                        default=12, type=int)

    parser.add_argument("-r", "--random-seed",
                        dest="seed",
                        help="RNG seed. Negative for random",
                        default=1, type=int)

    parser.add_argument("--test-every",
                        dest="test_every",
                        help="How many batches between tests.",
                        default=1500, type=int)

    parser.add_argument("--test-subset",
                        dest="test_subset",
                        help="A subset for the validation tests.",
                        default=100000, type=int)

    parser.add_argument("--test-batchsize",
                        dest="test_batchsize",
                        help="Batch size for computing the validation loss. This can be a bit bigger than the training batch size.",
                        default=64, type=int)

    parser.add_argument("--gradient-clipping",
                        dest="gradient_clipping",
                        help="Gradient clipping.",
                        default=1.0, type=float)

    parser.add_argument("--lr-warmup",
                        dest="lr_warmup",
                        help="Learning rate warmup.",
                        default=5000, type=int)

    parser.add_argument("--sample-length",
                        dest="sample_length",
                        help="Number of character to sample.",
                        default=600, type=int)

    parser.add_argument("--attention-type", dest="attention_type",
                        help="Which type of self-attention to use (default, gpt2, wide, narrow, relative)",
                        default="default", type=str)
    options = parser.parse_args()
    go(options)
