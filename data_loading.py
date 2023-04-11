from data_rnn import sort_reviews
import data_rnn
import torch.nn.utils.rnn as rnn_utils
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
    # for seq in x_train
    for i in range(0, len(x_train)):
        if i % seq_len == 0:
            batch = x_train[i:i+seq_len]
            batches.append(batch)
    return batches



# for i in batched_sorted_reviews[:5]:
#     print([data_rnn.i2w[idx] for idx in i]) 


def batch_reviews_by_instance_and_by_tokens(x_train, i2w, batch_size, seq_len):
    batches = []
    for i in range(0, len(x_train), batch_size):
        batch = x_train[i:i+batch_size]
        batch_words = []
        batch_token_count = 0
        
        for seq in batch:
            seq_words = [i2w[idx] for idx in seq]
            seq_token_count = len(seq_words)
            
            # Add the sequence to the current batch if it doesn't exceed the token count
            if batch_token_count + seq_token_count <= seq_len:
                batch_words.append(seq_words)
                batch_token_count += seq_token_count
            
            # Start a new batch if the current batch exceeds the token count
            else:
            #     print("First 5 of the batch:\n")
            #     print(batch_words[:5], "\n")
                batches.append(batch_words)
                batch_words = [seq_words]
                batch_token_count = seq_token_count
        
        # # Print the final batch if it's not empty
            if batch_words:
                batches.append(batch_words)
    return batches

batched_sorted_reviews = batch_reviews_by_instance_and_by_tokens(sorted_reviews, data_rnn.i2w, BATCH_SIZE, SEQUENCE_LENGTH)

max_len = max(len(review) for review in [batched_sorted_reviews])

def pad_sequence():
    padded_seqs = []
    for review in batched_sorted_reviews:
        padded_seq = rnn_utils.pad_sequence([torch.tensor(data_rnn.i2w)], 
                                            batch_first=True, 
                                            padding_value=0, 
                                            total_length=max_len)
        # padded_seq = review + [pad_token] * (max_len - len(review))
        padded_seqs.append(padded_seq)
    return padded_seqs

def pad_batch_reviews(batch_reviews, pad_token):
    # Get the maximum length of the sequences in the batch
    max_len = max(len(review) for review in batch_reviews)

    # Pad all sequences to the maximum length with the pad_token
    padded_batch = []
    for review in batched_sorted_reviews:
        
        padded_batch.append(padded_review)

    return padded_batch


padded_sequence = pad_sequence()






