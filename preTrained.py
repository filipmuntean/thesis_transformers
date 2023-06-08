from transformers import GPT2Tokenizer, GPT2LMHeadModel, modeling_utils
import os , gzip, torch, pandas as pd, fire, random, gc, wandb, math
from torch import nn
from basicTransformer import TransformerBlock
import torch.nn.functional as F
from collections.abc import Iterable

NUCLEUS_P = 0.9
TOP_P = 0.9
LOG2E = math.log2(math.e)
GRAD_CLIP = 1.0
DROPOUT = 0.0
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
PRINT_EVERY = 5 # How many batches between the printing sample
max_iters = 1000 # Number of batches to train on
PRINT_SEED_SIZE = 100 # The size of the sample seed
PRINT_SIZE = 600 # Number of tokens to print when printing sample 
SAMPLING_TEMP = 1.0 # Sampling temperature
TEST_EVERY = 1500 # How many batches between testing phases
checkpoint = None
IBLOCKS = 3
TEST_SUBSET = 1000
TEST_BATCH_SIZE = 4
DESC_CLIP = 2000
EPOCHS = 30
# '1315', 'Science & Medicine'
# '1304', 'Education'
# '1480', 'Software How-To'
# '1318', 'Technology'
# '1448', 'Tech News'
# '1477', 'Natural Sciences'
# '1468',  Educational Technology
PD_GENRES = [1315, 1304, 1480, 1318, 1448, 1477, 1468]

PD_TITLE_LENTGH = 100
###################################################
PD_SEED_LONG = """description: Ever wondered when you're at the grocery store, how it's possible that there are plastic bags with three bell pepper of exactly 500 grams? Or how the traffic light knows you are there and goes to green? Or, after a nice time out with friends, what the journey is of a Tikkie you send?
It is all about informatics. Artificial intelligence is taking over. But how smart is the computer? Is it also creative? For instance: can it create a title, logo and tune for this new show? We will find out in the first episode!

Other episodes:
- Can AI help solve the mystery what happened with the missing computer genius Jim Gray? How artificial intelligence helps to solve crimes.
- How do you get rich with computer science? The story of four smart students that earned millions with their new computer vision idea.
- What does the Efteling know about me? How facial recognition is integrated in our everyday lives.

Listen to this podcast and stumble upon captivating stories about humans and machines. Enter the fascinating world of computer science, algorithms, and artificial intelligence. 

title: """

PD_SEED = """description: Ever wondered when you're at the grocery store, how it's possible that there are plastic bags with three bell pepper of exactly 500 grams? Or how the traffic light knows you are there and goes to green? Or, after a nice time out with friends, what the journey is of a Tikkie you send?
It is all about informatics. Artificial intelligence is taking over. But how smart is the computer? Is it also creative? For instance: can it create a title, logo and tune for this new show? We will find out in the first episode!
Listen to this podcast and stumble upon captivating stories about humans and machines. Enter the fascinating world of computer science, algorithms, and artificial intelligence. 

title: """

# wandb.init(project="preTrained Generator on Cluster", entity="filipmuntean", config={"learning_rate": 3e-4, "batch_size": 32})

# Hyperparameters
gptname = 'gpt2'
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
        # return self.mod[0](x)

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

    def __init__(self, iblocks=3, gptname='gpt2', dropout=0.0, csize=None):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(gptname, encoder_hidden_states = True)
        model = GPT2LMHeadModel.from_pretrained(gptname, output_hidden_states=True)
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
        # x = self.model(x)[0]
        # x =  0.0 * cond.view(b, -1).sum(dim=1) #hack

        return x + self.headbias

    def clear(self):

        del self.container[0]
        del self.container
        self.container = [None]

        for block in self.iblocks:
            block.clear()

def contains_inf(input):
    if (not isinstance(input, torch.Tensor)) and isinstance(input, Iterable):
        for i in input:
            if contains_inf(i):
                return True
        return False
    else:
        return bool(torch.isinf(input).sum() > 0)

def contains_nan(input):
    if (not isinstance(input, torch.Tensor)) and isinstance(input, Iterable):
        for i in input:
            if contains_nan(i):
                return True
        return False
    else:
        return bool(torch.isnan(input).sum() > 0)


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

    if temperature != 1.0:
            lnprobs = lnprobs / temperature
    # Top-p/top-k filtering
    next_token_logits = modeling_utils.top_k_top_p_filtering(lnprobs[None, :], top_p=NUCLEUS_P)
    # Sample

    probs = F.softmax(next_token_logits, dim=-1)
    if contains_inf(probs) or contains_nan(probs):
        raise Exception(f'probs {probs}, logits {next_token_logits}')

    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

    return next_token


def enwik8(path):
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

    str_train, str_val, str_test = enwik8(data)
    str_train, str_test = (str_train + str_val, str_test) \
                            if final else (str_train, str_val)
    
    return str_train, str_test, str_val

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

def go():

    # if PD_SEED < 0:
    #     seed = random.randint(0, 1000000)
    #     print('random seed: ', seed)
    # else:
    #     torch.manual_seed(PD_SEED)

    str_train, str_test, str_val = load_data()
    
    # create the model
    model = GPT2Wrapper(iblocks = 3)

    if torch.cuda.is_available():
        model.to('cuda')
        model.model.mod[0].to('cuda')

    # tokenize the data
    data_train, data_val, data_test = \
        torch.tensor(model.tokenizer.encode(str_train)), \
        torch.tensor(model.tokenizer.encode(str_val)), \
        torch.tensor(model.tokenizer.encode(str_test))

    opt = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
    # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
    # -- linear learning rate warmup

    # training loop
    # -- note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    for i in range(max_iters):

        opt.zero_grad()

        # sample a batch of random subsequences
        starts = torch.randint(size=(BATCH_SIZE, ), low=0, high=data_train.size(0) - model.ctx - 1)
        seqs_source = [data_train[start  :start+model.ctx  ] for start in starts]
        seqs_target = [data_train[start+1:start+model.ctx+1] for start in starts]

        source = torch.cat([s[None, :] for s in seqs_source ], dim=0).to(torch.long)
        target = torch.cat([s[None, :] for s in seqs_target ], dim=0).to(torch.long)
        # -- target is the same sequence as source, except one character ahead

        if torch.cuda.is_available():
            source, target = source.to('cuda'), target.to('cuda')

        output = model(source)

        loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')

        loss.backward()

        # clip gradients
        # - If the total gradient vector has a length > 1, we clip it back down to 1.
        if GRAD_CLIP > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        opt.step()
        # sch.step()

        model.clear()

        # - validate every {arg.test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        if i != 0 and (i % PRINT_EVERY == 0 or i == max_iters - 1):

            with torch.no_grad():

                # generate and print some random text
                seedfr = random.randint(0, data_test.size(0) - PRINT_SEED_SIZE)
                input = data_test[seedfr:seedfr + PRINT_SEED_SIZE].to(torch.long)

                if torch.cuda.is_available():
                    input = input.cuda()

                # print the seed
                strinput = model.tokenizer.decode(input)
                print(f'[{strinput}]', end='')

                outseq = []
                for _ in range(PRINT_SIZE):
                    output = model(input[None, :])
                    c = sample(output[0, -1, :], SAMPLING_TEMP)
                    outseq.append(c[None])

                    input = torch.cat([input[1:], c[None]], dim=0)

                outseq = torch.cat(outseq, dim=0)
                outseq = model.tokenizer.decode(outseq)

                print(outseq)
        
        # val
        if i != 0 and (i % TEST_EVERY == 0 or i == max_iters - 1):

            with torch.no_grad():

                upto = data_test.size(0) if i == max_iters - 1 else TEST_SUBSET
                data_sub = data_test[:upto]

                bits, tot = 0.0, 0
                batch = [] # buffer, every time it fills up, we run it through the model

                for current in range(data_sub.size(0)):

                    fr = max(0, current - model.ctx)
                    to = current + 1

                    context = data_sub[fr:to].to(torch.long)
                    if context.size(0) < model.ctx + 1:
                        pad = torch.zeros(size=(model.ctx + 1 - context.size(0),), dtype=torch.long)
                        context = torch.cat([pad, context], dim=0)

                        assert context.size(0) == model.ctx + 1

                    if torch.cuda.is_available():
                        context = context.cuda()

                    batch.append(context[None, :])

                    if len(batch) == TEST_BATCH_SIZE or current == data_sub.size(0) - 1:

                        # batch is full, run it through the model
                        b = len(batch)

                        all = torch.cat(batch, dim=0)
                        source = all[:, :-1] # input
                        target = all[:, -1]  # target values

                        output = model(source)

                        lnprobs = output[torch.arange(b, device = d()), -1, target]
                        log2probs = lnprobs * LOG2E # convert from nats to bits

                        bits += - log2probs.sum()
                        batch = [] # empty buffer

                bits_per_byte = bits / data_sub.size(0)

                # print validation performance. 0.92 bit per byte is (currently) state of the art.
                print(f'epoch{i}: {bits_per_byte:.4} bits per byte')

def tobatch(df, tokenizer, g2i, normalize_genres=True, limit=2000, glist=None):

    with torch.no_grad(): # just in case

        # batch of n-hot vectors for genres
        ng = len(g2i)
        genres = torch.zeros(len(df), ng)

        for row in range(len(df)):
            dfgs = [int(g) for g in eval(df.iloc[row]['Genre IDs'])]
            if len(dfgs) == 0:
                print(eval(df.iloc[row]['Genre IDs']))
                raise Exception('Encountered podcast without genres.')
            for intg in dfgs:
                genres[row, g2i[intg]] = 1

        if normalize_genres:
            genres = genres / genres.sum(dim=1, keepdim=True)

        # batch of tokenized text
        strings = []

        for row in range(len(df)):
            name = df.iloc[row]['Name']
            desc = str(df.iloc[row]['Description'])[:limit]

            desc = desc.replace('\n', '')

            if glist is not None:
                dfgs = [int(g) for g in eval(df.iloc[row]['Genre IDs'])]
                dfgs = ', '.join([glist[ig] for ig in dfgs])

                strings.append(f'description: {desc} \ngenres: {dfgs} \ntitle: {name} ||\n')
            else:
                strings.append(f'description: {desc} \ntitle: {name} ||\n')

        ids = []
        for string in strings:
            ids.append(tokenizer.encode(string))

        # pad to max
        mx = min(max([len(id) for id in ids]), 1000)
        ids = [id[:mx] + ( [0] * (mx - len(id)) ) for id in ids]

        # I think zero should work as a pad token
        ids = [torch.tensor(id)[None, :] for id in ids]
        ids = torch.cat(ids, dim=0)

    # print([ (g2i[int(strid)],strid) for strid in eval(df.iloc[row]['Genre IDs'])])
    # print([(g2i[id], name) for id, name in glist.items()])
    # print(dfgs)
    # print(genres)

    return ids, genres


def go_pods():

    # if arg.seed < 0:
    #     seed = random.randint(0, 1000000)
    #     print('random seed: ', seed)
    # else:
    #     torch.manual_seed(arg.seed)

    df = pd.read_csv(here('mmi349/thesis_transformers/data/df_popular_podcasts.csv'))
    # data = here('/home/mmi349/thesis_transformers/data/enwik8.gz')
    with open(here('mmi349/thesis_transformers/data/genre_IDs.txt')) as file:
        glist = eval(file.read())
        glist = {int(idstr) : name for (idstr, name) in glist}
        rlist = {name : id for (id, name) in glist.items()}

    gs = set()
    for genres in df['Genre IDs']:
        genres = eval(genres)
        for genre in genres:
            g = int(genre)
            gs.add(g)

    i2g = list(gs)
    g2i = {g:i for i, g in enumerate(i2g)}

    train, val, test = df.iloc[:8000], df.iloc[8000:9000], df.iloc[9000:]

    # create the model
    model = GPT2Wrapper(iblocks=IBLOCKS, csize=len(i2g), gptname=gptname)

    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=torch.device('cpu')))

    if torch.cuda.is_available():
        model.to('cuda')
        model.model.mod[0].to('cuda')

    tok = model.tokenizer
    opt = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
    # sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / (arg.lr_warmup / arg.batch_size), 1.0))
    # -- linear learning rate warmup

    # training loop
    # -- note: we don't loop over the data, instead we sample a batch of random subsequences each time.
    seen = 0
    for e in range(EPOCHS):

        if e % PRINT_EVERY == 0:
            with torch.no_grad():

                # Generate 10 titles from the seed
                genres = torch.zeros(1, len(i2g))
                for genre in PD_GENRES:
                    # print(glist[genre])
                    genres[0, g2i[genre]] = 1.0

                for i in range(10):

                    # generate and print some random text
                    seed = PD_SEED
                    input = torch.tensor(tok.encode(seed))

                    if torch.cuda.is_available():
                        input, genres = input.to('cuda'), genres.to('cuda')

                    outseq = []
                    for _ in range(PD_TITLE_LENTGH):
                        output = model(input[None, :], cond=genres)
                        c = sample(output[0, -1, :], SAMPLING_TEMP)
                        outseq.append(c)

                        input = torch.cat([input, c], dim=0)

                    outseq = torch.cat(outseq, dim=0)
                    outseq = model.tokenizer.decode(outseq)

                    with open(f'pd.e{e:03}i{i:02}.txt', 'w') as file:
                        print(outseq[len(PD_SEED):], file=file)
                        print('---------------------------------------------\n', file=file)

                        print(PD_SEED + outseq, file=file)
        
    # Generate 10 random podcasts
                for i in range(10):
                    # generate a random genre
                    random_genre = random.choice(list(glist.keys()))

                    genres = torch.zeros(1, len(i2g))
                    genres[0, g2i[random_genre]] = 1.0

                    # generate and print some random text
                    seed = 'description: '
                    input = torch.tensor(tok.encode(seed))

                    if torch.cuda.is_available():
                        input, genres = input.to('cuda'), genres.to('cuda')

                    outseq = []
                    for _ in range(PRINT_SIZE):
                        output = model(input[None, :], cond=genres)
                        c = sample(output[0, -1, :], SAMPLING_TEMP)
                        outseq.append(c)

                        input = torch.cat([input, c], dim=0)

                    outseq = torch.cat(outseq, dim=0)
                    outseq = model.tokenizer.decode(outseq)

                    with open(f'random.e{e:03}i{i:02}.txt', 'w') as file:
                        print('chosen genre ', glist[random_genre], file=file)
                        print('---------------------------------------------', file=file)
                        print(seed, file=file)
                        print(outseq, flush=True, file=file)

        for fr in range(0, len(train), BATCH_SIZE):

            to = min(len(train), fr+BATCH_SIZE)

            dfbatch = df.iloc[fr:to]
            texts, genres = tobatch(dfbatch, tok, g2i, limit=DESC_CLIP, glist=glist)

            b = texts.size(0)
            source = torch.cat([torch.empty(b, 1, dtype=torch.long).fill_(0), texts], dim=1)
            target = torch.cat([texts, torch.empty(b, 1, dtype=torch.long).fill_(0)], dim=1)

            seen += b

            opt.zero_grad()

            if DROPOUT > 0.0:
                source = source * torch.empty_like(source).bernoulli_(DROPOUT)
                #-- word dropout on the input (help the model use the conditionals)

            if torch.cuda.is_available():
                source, target, genres = source.to('cuda'), target.to('cuda'), genres.to('cuda')

            output = model(source, cond=genres)

            loss = F.cross_entropy(output.transpose(2, 1), target, reduction='mean')

            loss.backward()

            # clip gradients
            # - If the total gradient vector has a length > 1, we clip it back down to 1.
            if GRAD_CLIP > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()
            # sch.step()

            del loss, source, target, genres
            model.clear()

            # for obj in gc.get_objects():
            #     try:
            #         if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            #             if obj.size(0) == b:
            #                 print(type(obj), obj.size())
            #     except:
            #         pass

        torch.save(model.state_dict(), './checkpoint.model')

        # - validate every {arg.test_every} steps. First we compute the
        #   compression on the validation (or a subset)
        #   then we generate some random text to monitor progress
        # if e != 0 and (e % arg.print_every == 0 or e == arg.epochs - 1):

        print('multipliers:')
        for block in model.iblocks:
            print('    ', block.mult)
        print()

if __name__ == "__main__":
    # fire.Fire(go)
    go_pods()
