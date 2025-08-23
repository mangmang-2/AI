1. 파이썬설치
알아서..
2.  가상환경
mkdir my-llm
cd my-llm
python -m venv venv
3. 가상환경 실핼
venv\Scripts\activate
4. 라이브러리 설치
pip install transformers datasets accelerate

- https://pytorch.org/get-started/locally/#windows-pip (쿠다 12.6선택)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
5. 학습코드
train.py
6. 실행 결과(my_cpp_model 폴더 생성)

| 파일명                                     | 설명                                                          |
| --------------------------------------- | ----------------------------------------------------------- |
| `config.json`                           | 모델 구조 및 설정 정보 (ex. hidden size, num layers, vocab size 등)   |
| `pytorch_model.bin`                     | **학습된 모델의 가중치** 파일. 실제 "배운 내용"이 여기에 저장됨                     |
| `special_tokens_map.json`               | 토크나이저에서 사용되는 특수 토큰 정보 (e.g. `<pad>`, `<eos>`)               |
| `tokenizer_config.json`                 | 토크나이저 설정 (e.g. lower\_case, padding 방식 등)                   |
| `vocab.json` (또는 `merges.txt`)          | 토크나이저의 vocabulary 정보. GPT2는 `vocab.json` + `merges.txt` 두 개 |
| `added_tokens.json` (있을 수도 있고 없을 수도 있음) | 사용자 정의 토큰이 있을 경우 저장됨                                        |

만약 다른 컴퓨터에서 학습한 데이터를 사용하고 싶다면 여기있는걸 가져가야함

학습이 드럽게 안되어서 다른 모델 전환
pip install transformers accelerate bitsandbytes

생성
generate.py



