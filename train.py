from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# 모델/토크나이저 불러오기
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # 패딩 토큰 지정


# ✅ 여기서 파일 길이 체크 (학습용 데이터가 실제 있는지 확인)
with open("cpp_data.txt", "r", encoding="utf-8") as f:
    text = f.read()
    print("파일 길이:", len(text))  # 이게 0이면 학습 불가

# 학습 데이터 로딩 (.txt로 변경!)
dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="cpp_data.txt",  # 여기 중요
    block_size=32
)

# 데이터 콜레이터
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    save_total_limit=2
)

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 학습 시작
trainer.train()

# 모델 저장
trainer.save_model("my_cpp_model")
tokenizer.save_pretrained("my_cpp_model")

print("✅ 학습 완료! 모델이 'my_cpp_model' 폴더에 저장됐어.")
