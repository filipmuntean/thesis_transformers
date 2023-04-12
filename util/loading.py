import util.data_rnn as data_rnn
import torch
import torch.nn.utils.rnn as rnn_utils
from torch import nn

BATCH_SIZE = 32
SEQUENCE_LENGTH = 10

# Sort the reviews
def sort_reviews(i2w, x_train): 
    return sorted(x_train, key = lambda x: sum(len(i2w[w]) for w in x))

# Batch the reviews by instance
def batch_sequences_by_instance(sequence, batch_size):
    batches = []
    for i in range(0, len(sequence), batch_size):
        batch = sequence[i:i + batch_size]
        batch_words = [] 
        for seq in batch:
        #    words = [i2w[idx] for idx in sequence]
            batch_words.append(seq)
        batches.append(batch_words)

    return batches

def batch_sequences_by_tokens(sequence, batch_size, seq_len):
    # Sort the data by the total number of tokens
    for i in range(0, len(sequence), batch_size):
        # Batch the sequences by the maximum number of tokens
        batch = sequence[i:i+batch_size]
        current_batch = []
        current_batch_tokens = 0
        for seq in sequence:
            sequence_tokens = sum(seq)
            
            # If the current sequence exceeds the maximum number of tokens, skip it
            if sequence_tokens > seq_len:
                continue
            
            # If the current sequence can fit into the current batch, add it
            if current_batch_tokens + sequence_tokens <= seq_len:
                current_batch.append(seq)
                current_batch_tokens += sequence_tokens
            
            # If the current sequence cannot fit into the current batch, start a new batch
            else:
                batch.append(current_batch)
                current_batch = seq
                current_batch_tokens = sequence_tokens
        
        # Add the final batch if it's not empty
        if current_batch:
            batch.append(current_batch)
    return batch


def get_max(sorted_sequence):
    max_len = max(len(review) for review in sorted_sequence)
    # max_len = max(map(len, sorted_sequence))
    return max_len

def get_padded_sequence(sequences):
    seq = []
    
    for batch in sequences:
        max_batch = get_max(batch)
        for seq in batch:
            if max_batch is None:
                max_batch = get_max(sequences)
            if len(seq) > max_batch:
                seq = seq[:max_batch]
            # Pad sequence to max length with zeros
            seq += [0] * (max_batch - len(seq))
            # print(f"Sequence length: {len(seq)}, Max length: {max_len}")
    return sequences

def get_tensor(padded_sequence):
    for batch in padded_sequence:
        for seq in batch: 
            padded_tensor = torch.tensor(seq, dtype = torch.long)
    return padded_tensor

## TODO use fire module

## Simple self attention
## Each of the outputs is a weighted vector of the input vectors
## Sum of the square sequence lengths is a constant over the tokens







