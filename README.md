# counseling-data

한국어 심리상담 대화 데이터셋을 **영어로 번역·정제**하고, 학습용 **CSV(train/valid)** 로 만드는 전처리 스크립트 모음입니다.
번역은 로컬 LLM(Ollama, `qwen3:14b`)을 사용합니다.

> 데이터 파일(`.txt`, `.json`, `.csv`)은 저장소에 포함되어 있지 않습니다. 스크립트만 있습니다.

## 데이터셋

| 데이터셋 | 내용 | 라벨 |
|---|---|---|
| `counseling_dataset` | 일반 상담 대화 (`resource_X.txt` ↔ `label_X.json`) | 우울·불안·중독 점수, 나이, 성별, 요약 등 |
| `adolescent_consultant_dataset` | 청소년 상담 대화 (`inputs/{유형}/0008.txt` ↔ `labels/{유형}/0008.json`) | 위기단계, 합계점수, 학대의심, 임상가 소견 등 |

청소년 데이터의 유형은 `정상군 / 관찰필요 / 상담필요 / 응급 / 학대의심` 5개입니다.

## 처리 흐름

```
한국어 원본 ──translate_to_english.py──▶ 영어 번역본 ──(정제 스크립트)──▶ build_*_csv.py ──▶ train.csv / valid.csv
```

## 주요 코드

### 공통: 번역 — `translate_to_english.py`
- 한국어 `.txt` 대화를 영어로 번역하고, 짝이 되는 `.json` 라벨의 텍스트 필드도 함께 번역합니다. 숫자·ID·범주형 값은 그대로 둡니다.
- 긴 대화는 줄마다 `[N]` 번호를 붙여 청크로 나눠 번역합니다. 결과를 번호로 다시 맞추고, 앞 청크의 마지막 20줄을 문맥으로 넘겨 번역이 자연스럽게 이어지게 합니다.
- 줄 수가 맞지 않으면 다시 시도하고, 계속 실패하면 청크를 반으로 나눕니다. 번역 결과에 한글 등 비영어 문자가 남아 있으면 다시 번역합니다.
- 여러 파일을 병렬로 처리하고, 이미 번역된 파일은 건너뜁니다. 실패한 파일은 `error_log.json`에 기록합니다.

### `counseling_dataset/`
| 파일 | 기능 |
|---|---|
| `korean_counseling_dataset/translate_to_english.py` | 대화와 `paragraph_text`, `summary`를 영어로 번역 |
| `korean_counseling_dataset/scan_sizes.py` | 대화 파일 길이(문자 수, 추정 토큰 수) 통계 → LLM context 크기 설정에 참고 |
| `korean_counseling_dataset/build_korean_csv.py` | 한국어 세션 1개 = 1행으로 `korean_train.csv` / `korean_valid.csv` 생성 |
| `english_counseling_dataset/find_non_english.py` | 번역본에 비영어 문자가 남은 파일과 줄 번호를 찾아 `error_log.json`에 기록 |
| `english_counseling_dataset/build_english_csv.py` | 영어 번역본으로 `english_train.csv` / `english_valid.csv` 생성 |

### `adolescent_consultant_dataset/`
| 파일 | 기능 |
|---|---|
| `korean_.../translate_to_english.py` | 대화와 `임상가 종합소견`, `임상가코멘트`를 영어로 번역 |
| `korean_.../correct_english_dataset.py` | 상담사 발화의 1인칭 대명사(I/me/my)를 2인칭(you/your)으로 LLM으로 교정 → `..._corrected/` 생성 |
| `english_.../sample_check.py` | 교정본에서 TXT와 JSON이 어긋나는 사례(화자 뒤바뀜, 대명사, 줄 밀림)를 점검 |
| `english_.../fix_english_dataset.py` | LLM 없이 규칙으로 TXT와 JSON 불일치 5종을 수정 → `post-train/`, `post-valid/` 생성 |
| `fix_audio_chunks.py` | 중복 제거 과정에서 빠진 `audio` 구간을 한국어 원본의 타임스탬프 기준으로 복원 |
| `*/build_adolescent_csv.py` | 세션 1개 = 1행 (JSON `info` 필드 + 유형 + 대화 전문)으로 train/valid CSV 생성 |

## 실행

```bash
pip install ollama tqdm pandas
ollama pull qwen3:14b
OLLAMA_NUM_PARALLEL=4 ollama serve   # 병렬 번역 시

python counseling_dataset/korean_counseling_dataset/translate_to_english.py
python counseling_dataset/english_counseling_dataset/build_english_csv.py
```

각 스크립트는 자기 위치를 기준으로 `train/`, `valid/` 폴더를 찾습니다.
단, `fix_audio_chunks.py`와 `sample_check.py`는 경로가 Windows 절대경로로 고정되어 있으니, 실행 전에 스크립트 상단의 `BASE` / `base` 값을 수정하세요.
