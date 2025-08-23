from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "EleutherAI/gpt-neo-1.3B"  # ✅ 로그인 없이 바로 사용 가능

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "int sum(int a, int b) {\n    return"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=50,
    do_sample=True,
    top_k=50,
    temperature=0.7,
    max_new_tokens=128,
    num_return_sequences=1
)

print("🧠 결과:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
