import util
from main import Main

def get_test_values():

    sorted_x_val, sorted_y_val = util.loading.sort_reviews(Main.i2w, Main.x_val, Main.y_val)

    batched_x_val, batched_y_val = util.loading.batch_sequences_by_instance(sorted_x_val, sorted_y_val, batch_size = 32)

    padded_reviews_test_x, padded_sentiments_test_y = util.loading.get_padded_sequences_and_labels(batched_x_val, batched_y_val)
    # Concatenate the test tensors
    test_reviews = util.loading.get_review_tensor(padded_reviews_test_x)

    test_sentiments = util.loading.get_sentiment_tensor(padded_sentiments_test_y)

    test_instance_dataset = util.loading.append_lists(test_reviews, test_sentiments)

get_test_values