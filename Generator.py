import util
# from util import Transformer
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Transformer import transformer
from main import Main

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def testTransformer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 100
    trg_vocab_size = 100
    model = transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    for (src, trg) in enumerate(util.Main.train_dataset):
        trg_flat = list(itertools.chain(*trg))
        src_tensor = torch.tensor(src, dtype=torch.long).to(device)
        trg_tensor = torch.tensor(trg_flat, dtype=torch.long).to(device)
        out = model(src_tensor, trg_tensor[:-1])
        print(out.shape)

# def generate_random_batch(data, batch_size, seq_length):
#     """
#     Generate a batch of training examples by randomly sampling
#     subsequences from the data tensor.

#     :param data: A tensor of shape (n_samples, max_seq_length)
#     :param batch_size: The number of samples in a batch
#     :param seq_length: The length of each subsequence to sample
#     :return: A tuple of two tensors:
#                 - input_batch: A tensor of shape (batch_size, seq_length)
#                 - target_batch: A tensor of shape (batch_size, seq_length)
#     """
#     # randomly sample batch_size start positions from the data tensor
#     starts = torch.randint(size=(batch_size,), low=0, high=data.size(1) - seq_length - 1)

#     # create a list of subsequences of length seq_length from the data tensor
#     subsequences = [data[:, start:start+seq_length] for start in starts]

#     # stack the list of subsequences into a tensor of shape (batch_size, seq_length)
#     input_batch = torch.stack(subsequences, dim=0)

#     # create a target batch by shifting the input batch by one position
#     target_batch = torch.roll(input_batch, shifts=-1, dims=1)

#     return input_batch, target_batch

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


    starts = []
    # Sample the starting indices of the sequences to slice out for each tensor in the list
    for tensor in data:
        starts.append(torch.randint(size=(batch_size,), low=0, high=max(tensor.size(0) - length - 1, 1)))

    # Slice out the input sequences and targets for each tensor in the list
    seqs_inputs = [torch.stack([tensor[starts[i][j]:starts[i][j] + length] for j in range(batch_size)]) for i, tensor in enumerate(data)]
    seqs_targets = [torch.stack([tensor[starts[i][j] + 1:starts[i][j] + length + 1] for j in range(batch_size)]) for i, tensor in enumerate(data)]

    # seqs_inputs = [torch.cat([tensor[start:start + length], torch.zeros(int(max_length - length), dtype=torch.long)]) for tensor, start in zip(data, starts)]
    # seqs_targets = [torch.cat([tensor[start + 1:start + length + 1], torch.zeros(int(max_length - length), dtype=torch.long)]) for tensor, start in zip(data, starts)]
 
    max_length = 2514  # set to the maximum size in the last dimension

    padded_seqs_inputs = [F.pad(seq, pad=(0, 0, 0, max_length - seq.shape[-1]), mode='constant', value=0) for seq in seqs_inputs]
    padded_seqs_targets = [F.pad(seq, pad=(0, 0, 0, max_length - seq.shape[-1]), mode='constant', value=0) for seq in seqs_targets]
    # Concatenate the sequences for all tensors into single input and target tensors
    inputs = torch.cat([s[None, :] for s in padded_seqs_inputs], dim=0).to(torch.long).to(device)
    targets = torch.cat([s[None, :] for s in padded_seqs_targets], dim=0).to(torch.long).to(device)

    return inputs, targets
    
    
data = Main.review_tensors
num_epochs = 3
batch_size = 32
seq_length = 65

for i in range(num_epochs):
    # generate a batch of training examples
    input_batch, target_batch = sample_batch(Main.review_tensors, length=12, batch_size=16)

        # forward pass, backward pass, and update weights here
    # # Step 4: Create an instance of the transformer and move it to the GPU
    # device = torch.device("meta" if torch.cuda.is_available() else "cpu")
    # transformer = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(transformer.parameters(), lr=0.001)

    # # Step 5: Train the transformer
    # for epoch in range(epochs):
        
    #         src = src.to(device)
    #         trg = trg.to(device)

    #         optimizer.zero_grad()

    #         output = transformer(src, trg[:, :-1])
    #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

    #         loss.backward()
    #         optimizer.step()

    # # Step 6: Evaluate the transformer on the test set
    # transformer.eval()
    # with torch.no_grad():
    #     for i, (src, trg) in enumerate(Main.test_dataset):
    #         src = src.to(device)
    #         trg = trg.to(device)

    #         output = transformer(src, trg[:, :-1])
    #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

    # # Step 7: Save the model
    # torch.save(transformer.state_dict(), "transformer.pth")

testTransformer()

        

