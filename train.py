from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# 모델과 토크나이저 로딩 (GPT-Neo 125M)
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# pad_token 설정 (없을 경우 오류 방지용)
tokenizer.pad_token = tokenizer.eos_token

# 학습 데이터 로딩
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="cpp_data.txt",  # ★ 학습 데이터 텍스트 파일 (JSONL 아님 주의)
    block_size=64              # 32, 64, 128 가능. GPU VRAM 따라 조절
)

# MLM은 아님 (GPT 계열은 causal LM)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./my_neo_model",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=5,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 파일 길이 확인용
with open("cpp_data.txt", "r", encoding="utf-8") as f:
    text = f.read()
    print("📝 학습 텍스트 길이:", len(text))

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("my_neo_model")
tokenizer.save_pretrained("my_neo_model")

print("✅ GPT-Neo 학습 완료! my_neo_model 폴더에 저장됐어.")
