import util
import torch

class Main():
    
    # Load the data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_reviews = util.loading.sort_reviews(i2w, x_train)

    # Batch by a number of fixed instances
    batched_sorted_reviews = util.loading.batch_sequences_by_instance(sorted_reviews, batch_size = 32)
    
    # Get the maximum length for padding
    # max_len = util.loading.get_max(batched_sorted_reviews)

    # Pad reviews manually with 0's
    padded_reviews = util.loading.get_padded_sequence(batched_sorted_reviews)
    
    # Build tensors
    padded_tensors = util.loading.get_tensor(padded_reviews)

main = Main()








