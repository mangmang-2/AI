from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("my_cpp_model")
tokenizer = GPT2Tokenizer.from_pretrained("my_cpp_model")

prompt = "int multiply(int a, int b) {"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.8,
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
