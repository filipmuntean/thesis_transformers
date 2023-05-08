import util
# from util import Transformer
import torch
import itertools
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Transformer import transformer
from main import Main
import random
from torch.nn.utils.rnn import pad_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def testTransformer():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     src_pad_idx = 0
#     trg_pad_idx = 0
#     src_vocab_size = 100
#     trg_vocab_size = 100
#     model = transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

#     for (src, trg) in enumerate(util.Main.train_dataset):
#         trg_flat = list(itertools.chain(*trg))
#         src_tensor = torch.tensor(src, dtype=torch.long).to(device)
#         trg_tensor = torch.tensor(trg_flat, dtype=torch.long).to(device)
#         out = model(src_tensor, trg_tensor[:-1])
#         print(out.shape)

# def exemplify():
#     block_size = 8
#     # x = util.Main.review_tensors[:block_size]
#     # y = util.Main.review_tensors[1:block_size + 1]

#     review_counter = 0
#     for tensor in Main.review_tensors[:block_size]:
#         for review in tensor:
#             print(review)
#             # if review_counter == 33:
#             #     for i in range(len(review)-1):
#             #         input_tokens = review[:i+1]
#             #         target_token = review[i+1]
#             #         print(f"When input is tensor {input_tokens}, the target is: {target_token}")
#             #         if review[i] == 0 and review[i+1] == 0:
#             #             break  # stop looping through this review
#             #     # Increment review_counter only after all tokens in the review have been printed
#             # review_counter += 1
# exemplify()

torch.manual_seed(1337)
batch_size = 4
block_size = 8

def get_batch():
    data = Main.review_tensors
    batch = random.choice(data)
    data_size = len(data)
    tensor_size = batch.size(0)
    ix = torch.randint(tensor_size - block_size, (batch_size,))
    x_list, y_list = [], []
    for tensor in data[:1]:
        for review in tensor:
            for i in range(len(review)-1):
                x = torch.stack([review[i:i+block_size] for i in ix])
                y = torch.stack([review[i+1:i+block_size+1] for i in ix])
                
            x_list.append(x)
            y_list.append(y)
            # x = random.choice(x_list)
            # y = random.choice(y_list)
            
    # return x, y

    # for tensor in data:
    #     for review in tensor:
    #         tensor_size = tensor.size(0)
    #         ix = torch.randint(tensor_size - block_size, (batch_size,))
    #         for i in range(len(review)-1):
                    
    #             x = torch.stack([tensor[i:i+block_size] for i in ix])
    #             y = torch.stack([tensor[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch()
print('inputs:')
print(xb.shape, xb)
print('\ntargets:')
print(yb.shape, yb)
print('-------')

for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t + 1]
        target = yb[b, t]
        print(f"when input is {context} the target is {target}")

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

# def sample_batch(data, length, batch_size):
#     """
#     Takes the data (a single sequence of tokens) and slices out a batch of subsequences to provide as input to the model.

#     For each input instance, it also slices out the sequence that is shofted one position to the right, to provide as a
#     target for the model.

#     :param data: The (training) data. A single vector of tokens represented by integers
#     :param length: The length of the subsequences in the batch.
#     :param batch_size: The number of subsequences in the batch
#     :return: A pair (input, target) of minteger matrices representing the input and target for the model.
#     """

#     batch_indices = random.sample(range(len(data)), batch_size)

#     max_size = max([data[i].size(1) for i in batch_indices])
#     # padded_tensors = []
#     # for i in batch_indices:
#     #     seqs = data[i]
#     #     max_size = max(max_size, seqs.size(1))
#     #     for seq in seqs:
#     #         padded_tensors.append(torch.nn.functional.pad(seq, pad, mode='constant'))
#     #     batches.append(len(padded_tensors))
#     padded_tensors = []
#     batches = []
#     for i in range(0, len(data), batch_size):
#         batch = data[i:i + batch_size]
#         for seq in batch:
#             pad = (0, max_size - seq.size(1))
#             # pad = (0, max_size - len(seq))
#             padded_tensors.append(torch.nn.functional.pad(seq, pad, mode='constant'))
#         # batches.append(len(padded_tensors))

#     # stack the padded tensors along the batch dimension
#     padded_inputs = [padded_tensors[start:start + length] for start in range(0, len(padded_tensors) - length, length)]
#     padded_targets = [padded_tensors[start + 1:start + length + 1] for start in range(0, len(padded_tensors) - length, length)]
    
#     inputs = torch.stack([s.unsqueeze(0) for s in padded_inputs], dim=0).to(torch.long)
#     targets = torch.stack([s.unsqueeze(0) for s in padded_targets], dim=0).to(torch.long)
#     return inputs, targets
    
   
 
# for i in range(num_epochs):
#     # generate a batch of training examples
#     input_batch, target_batch = sample_batch(data, length=4, batch_size=8)

#         # forward pass, backward pass, and update weights here
#     # # Step 4: Create an instance of the transformer and move it to the GPU
#     # device = torch.device("meta" if torch.cuda.is_available() else "cpu")
#     # transformer = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx).to(device)

#     # criterion = nn.CrossEntropyLoss()
#     # optimizer = optim.Adam(transformer.parameters(), lr=0.001)

#     # # Step 5: Train the transformer
#     # for epoch in range(epochs):
        
#     #         src = src.to(device)
#     #         trg = trg.to(device)

#     #         optimizer.zero_grad()

#     #         output = transformer(src, trg[:, :-1])
#     #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

#     #         loss.backward()
#     #         optimizer.step()

#     # # Step 6: Evaluate the transformer on the test set
#     # transformer.eval()
#     # with torch.no_grad():
#     #     for i, (src, trg) in enumerate(Main.test_dataset):
#     #         src = src.to(device)
#     #         trg = trg.to(device)

#     #         output = transformer(src, trg[:, :-1])
#     #         loss = criterion(output.reshape(-1, output.shape[-1]), trg[:, 1:].reshape(-1))

#     # # Step 7: Save the model
#     # torch.save(transformer.state_dict(), "transformer.pth")

# # sample_batch(data, length=4, batch_size=8)

        

