import util

class Tokens():

    # Load the data
    (x_train_tokens, y_train_tokens), (x_val_tokens, y_val_tokens), (i2w_tokens, w2i_tokens), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_x_train_tokens, sorted_y_train_tokens = util.loading_tokens.sort_reviews(i2w_tokens, x_train_tokens, y_train_tokens)

    # Batch by a number of fixed tokens
    batched_x_train_by_tokens, batched_y_train_by_tokens = util.loading_tokens.batch_sequences_by_tokens(sorted_x_train_tokens, sorted_y_train_tokens, batch_size = 32, seq_len = 8388608)

    padded_reviews_x_by_tokens, padded_reviews_y_by_tokens = util.loading_tokens.get_padded_sequences_and_labels(batched_x_train_by_tokens, batched_y_train_by_tokens)

    # Build tensors
    review_tensors_by_tokens = util.loading_tokens.get_review_tensor(padded_reviews_x_by_tokens)
        
    sentiment_tensors_by_tokens = util.loading_tokens.get_sentiment_tensor(padded_reviews_y_by_tokens)
    
    # Concatenate the test tensors
    train_dataset_by_tokens = util.loading.append_lists(review_tensors_by_tokens, sentiment_tensors_by_tokens)
    
    # test_reviews = util.loading.get_review_tensor(x_val)
    # test_sentiments = util.loading.get_sentiment_tensor(y_val)

tokens = Tokens()
