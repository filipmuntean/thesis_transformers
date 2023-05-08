import util
import torch


class Main():

    # Load the data
    (x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = util.data_rnn.load_imdb(final = False)

    # Sort the reviews
    sorted_x_train, sorted_y_train = util.loading.sort_reviews(i2w, x_train, y_train)

    # Batch by a number of fixed instances
    batched_x_train, batched_y_train = util.loading.batch_sequences_by_instance(sorted_x_train, sorted_y_train, batch_size = 32)

    # Pad reviews manually with 0's. We also pad the labels so that they match the length of the batches of reviews
    padded_reviews_x, padded_sentiments_y = util.loading.get_padded_sequences_and_labels(batched_x_train, batched_y_train) 

    # Build tensors
    review_tensors = util.loading.get_review_tensor(padded_reviews_x)
    
    block_size = 8
    x = review_tensors[:block_size]
    y = review_tensors[1:block_size + 1]

    review_counter = 0
    for tensor in review_tensors[:block_size]:
        for review in tensor:
            if review_counter == 33:
                for i in range(len(review)-1):
                    input_tokens = review[:i+1]
                    target_token = review[i+1]
                    print(f"When input is tensor {input_tokens}, the target is: {target_token}")
                    if review[i] == 0 and review[i+1] == 0:
                        break  # stop looping through this review
                # Increment review_counter only after all tokens in the review have been printed
            review_counter += 1
    
    sentiment_tensors = util.loading.get_sentiment_tensor(padded_sentiments_y)
    
    train_dataset = util.loading.append_lists(review_tensors, sentiment_tensors)
    sorted_x_val, sorted_y_val = util.loading.sort_reviews(i2w, x_val, y_val)

    batched_x_val, batched_y_val = util.loading.batch_sequences_by_instance(sorted_x_val, sorted_y_val, batch_size = 32)

    padded_reviews_test_x, padded_sentiments_test_y = util.loading.get_padded_sequences_and_labels(batched_x_val, batched_y_val)
    # Concatenate the test tensors
    test_reviews = util.loading.get_review_tensor(padded_reviews_test_x)
    
    test_sentiments = util.loading.get_sentiment_tensor(padded_sentiments_test_y)

    test_dataset = util.loading.append_lists(test_reviews, test_sentiments)

main = Main()




 