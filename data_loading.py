from data_rnn import sort_reviews
import data_rnn
import torch

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

sorted_reviews = sort_reviews(data_rnn.i2w, data_rnn.x_train)

# for i in sorted_reviews[:5]:
#     print([data_rnn.i2w[idx] for idx in i]) 

def batch_reviews_by_instance(x_train, batch_size):
    batches = []
    for i in range(0, len(x_train), batch_size):
            batch = x_train[i:i+batch_size]
            # [i2w[idx] for idx in seq] for seq in batch]
            batches.append(batch)
    return batches

def batch_reviews_by_tokens(x_train, seq_len):
    batches = []
    for i in range(0, len(x_train)):
        if i % seq_len == 0:
            batch = x_train[i:i+seq_len]
            batches.append(batch)
    return batches

# def batch_tokens(x_train, seq_len):
#     batches = []
#     for seq in x_train:
#         for i in range(0, len(seq), seq_len):
#             batch = seq[i:i+seq_len]
#             batches.append([i2w[idx] for idx in batch])
#     return batches

# batched_sorted_reviews = batch_reviews_by_instance(sorted_reviews)

# for i in batched_sorted_reviews[:5]:
#     print([data_rnn.i2w[idx] for idx in i]) 

# print(batched_sorted_reviews[:5])

print(data_rnn.w2i['.pad']);


