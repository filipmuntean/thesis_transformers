from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "I am doing"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

last_hidden_state = output.last_hidden_state
print(last_hidden_state)
