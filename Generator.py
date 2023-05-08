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
    vocab_size = len(Main.i2w)
    tensor_size = batch.size(0)

    ix = torch.randint(tensor_size - block_size, (batch_size,))

    # for tensor in data[:batch_size + 1]:
    #     for review in data[:batch_size + 1]:
    #         for i in range(len(review)-1):
    #             input = torch.stack([review[i:i+block_size] for i in ix])
    #             target = torch.stack([review[i+1:i+block_size+1] for i in ix])
    input = torch.stack([review[ix[i]:ix[i] + block_size] for i, review in enumerate(batch)])
    target = torch.stack([review[ix[i] + 1:ix[i] + block_size + 1] for i, review in enumerate(batch)])

    return input, target

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

        

