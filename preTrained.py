from transformers import GPT2Tokenizer, GPT2LMHeadModel
from datasets import load_dataset
import os , gzip, torch
import numpy as np
import wandb, random, gc

wandb.init(project="preTrained Generator on Cluster", entity="filipmuntean", config={"learning_rate": 3e-4, "batch_size": 32})

# Hyperparameters
model_name = 'gpt2'
final = True
device = torch.device('cuda')
torch.cuda.empty_cache()
gc.collect()

# def enwik8(path, n_train=int(90e6), n_valid=int(5e6), n_test=int(5e6)):
#     """
#     This function was taken and adapted from Peter Bloem - Transformers from Scratch: https://github.com/pbloem/former/blob/master/experiments/generate.py
#     """
#     if path is None:
#         path = here('data/enwik8.gz')

#     with gzip.open(path) if path.endswith('.gz') else open(path) as file:
#         X = np.frombuffer(file.read(n_train + n_valid + n_test), dtype=np.uint8)
#         trX, vaX, teX = np.split(X, [n_train, n_train + n_valid])
#         train_dataset = np.copy(trX)
#         val_dataset = np.copy(vaX)
#         test_dataset = np.copy(teX)
#         return torch.from_numpy(train_dataset), torch.from_numpy(val_dataset), torch.from_numpy(test_dataset)
    
# def here(subpath=None):
#     """
#     This function was taken and adapted from Peter Bloem -  Transformers from Scratch: https://github.com/pbloem/former/blob/master/experiments/generate.py
#     """
#     if subpath is None:
#         return os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))

#     return os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', subpath))

# def load_data():
#     """
#     Load the enwik8 dataset from the Hutter challenge. This function was taken and adapted from Peter Bloem -  Transformers from Scratch: 
#     https://github.com/pbloem/former/blob/master/experiments/generate.py  """

#     data = here('/home/mmi349/thesis_transformers/data/enwik8.gz')
#     # data = here('filip/thesis/data/enwik8.gz')

#     data_train, data_val, data_test = enwik8(data) 
#     data_train, data_test = (torch.cat([data_train, data_val], dim=0), data_test) \
#                             if final == True else (data_train, data_val)

#     return data_train, data_test

# data_train, data_test = load_data()
# list_data_train = data_train.tolist()
# list_data_test = data_test.tolist()

dataset = load_dataset("enwik8")
print(dataset)

def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  encoding = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=1024)
  # For language modeling the labels need to be the input_ids
  #encoding["labels"] = encoding["input_ids"]
  return encoding

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

column_names = dataset["train"].column_names
dataset = dataset.map(encode_batch, remove_columns=column_names, batched=True)

block_size = 50
# Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
def group_texts(examples):
  # Concatenate all texts.
  concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
  total_length = len(concatenated_examples[list(examples.keys())[0]])
  # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
  # customize this part to your needs.
  total_length = (total_length // block_size) * block_size
  # Split by chunks of max_len.
  result = {
    k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
    for k, t in concatenated_examples.items()
  }
  result["labels"] = result["input_ids"].copy()
  return result

dataset = dataset.map(group_texts,batched=True,)

dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

model.to(device)
wandb.watch(model)
# decoded_data_test = [str(tokenizer.decode(item)) for item in list_data_test]

model_inputs = tokenizer.batch_encode_plus(
    decoded_data_test,
    add_special_tokens=True,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

input_ids = model_inputs["input_ids"]
attention_mask = model_inputs["attention_mask"]
input_ids = input_ids.to(device)
attention_mask = attention_mask.to(device)

def generate_text_from_dataset(dataset, max_length):
    with torch.no_grad():
        output = model.generate(
            input_ids=dataset.unsqueeze(0).to(device),
            attention_mask=attention_mask.unsqueeze(0).to(device),
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        output = output.to(device)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

prompt_length = 50  # The number of tokens to use as a prompt
max_length = 100  # The maximum length of the generated text
prompt = model_inputs["input_ids"][0][:prompt_length]   # Extract a prompt from the dataset

print(torch.cuda.memory_summary(device=None, abbreviated=False))
with torch.no_grad():
    generated_text = generate_text_from_dataset(prompt, max_length)
    gemerated_text = generated_text.to(device)
    wandb.log({"generated_text": generated_text})
    print(generated_text)

wandb.finish()
# text = "I am doing"

# encoded_input = tokenizer(data, return_tensors='pt')
# output = model(**encoded_input)

# last_hidden_state = output.last_hidden_state
# print(last_hidden_state)
