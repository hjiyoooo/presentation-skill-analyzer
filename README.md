ML term project
# 🎙️ Presentation Skill Analyzer

본 프로젝트는 한국어 강의 및 발표 데이터를 기반으로 발표 품질을 평가(Good/Poor)하고, 발표자에게 발표 분석 및 개선 피드백을 제공하는 ML 시스템이다.

## 프로젝트 개요
발표는 학업·취업·업무 전반의 핵심 역량이지만, 자신의 발화 습관을 혼자 객관적으로 파악하기 어렵습니다. 본 프로젝트는 AI Hub의 한국어 강의 음성 데이터(KLEC)를 활용해 필러워드 비율 기반 rule-based 레이블링 → 텍스트 + 수치형 피처 결합 분류 모델을 구축하고, Gradio 대시보드로 실시간 피드백을 제공합니다.
**핵심 기능**
- Audio STT Pipeline : OpenAI Whisper(medium)로 음성 파일을 한국어 전사
- Hybrid Classification : KLUE-RoBERTa [CLS] 벡터(768d) + 수치형 피처(5d) Concat → Good/Poor 이진 분류
- 항목별 피드백 : 필러워드 비율, 어휘 다양성(TTR), 말 속도(WPM) 3항목 수치 피드백
- Interactive Dashboard : Gradio 웹 인터페이스 (07_dashboard.ipynb)

## 프로젝트 구조
presentation-skill-analyzer/
├── notebooks/
│   ├── 01_EDA.ipynb                  # KsponSpeech 필러워드 분포 / 영어 평가 점수 분포 분석
│   ├── 02_preprocessing.ipynb        # KLEC 전처리, 피처 추출, 레이블링, 데이터 분할
│   ├── 03_baseline.ipynb             # Logistic Regression / Random Forest 베이스라인
│   ├── 04_main_model.ipynb           # KLUE-RoBERTa 단독 fine-tuning
│   ├── 05_feature_concat_model.ipynb # Feature Concat 모델 (class weight, Early Stopping)
│   ├── 05_feature_concat_model2.ipynb# 데이터 10K 확장 및 최종 실험
│   ├── 06_Inference.ipynb            # 저장 모델 로드 및 추론 파이프라인 검증
│   └── 07_dashboard.ipynb            # Gradio 대시보드
├── data/
│   ├── train.csv                     # 학습 데이터 (7,000개)
│   ├── val.csv                       # 검증 데이터 (1,000개)
│   └── test.csv                      # 테스트 데이터 (2,000개)
├── inference_utils.py                # 모델 정의, 피처 추출, 추론, 피드백 공통 유틸
├── requirements.txt
└── README.md

## 모델 아키텍처
[입력: 발화 전사 텍스트]
        │
   clean_text 정제
        │
  klue/roberta-small
        │
   [CLS] 벡터 (768d)
        │
   Dropout(0.1)         ← concat →   수치형 피처 (5d, StandardScaler 정규화)
        └──────────────────┬──────────────────┘
                     773d 결합 벡터
                           │
                   Linear(773 → 256)
                       ReLU
                   Dropout(0.1)
                   Linear(256 → 2)
                           │
               [Good(1) / Poor(0)] + 확률값


## 실험 결과 요약
모델              Accuracy    F1   PoorRecall   GoodRecall
-----------------------------------------------------------
Logistic 
Regression        62.1%    0.718      —             —
-----------------------------------------------------------
Random 
Forest            62.3%    0.679      —             —
-----------------------------------------------------------
KLUE-RoBERTa 
단독               59.9%    0.685     38%           76%
-----------------------------------------------------------
Concat 
(no weight)       64.1%    0.753      22%           95%
-----------------------------------------------------------
Concat 
(weight=2.0, ES)  71.5%    0.753      66%           76%
-----------------------------------------------------------
Concat 
(weight=2.5, ES)  69.3%    0.700      78%           63%

발표 약점 진단이 핵심 목적이므로 Poor(발화 품질 낮음)를 놓치지 않는 것이 중요합니다. Poor Recall 78%를 달성한 weight=2.5 + Early Stopping 모델을 최종 모델로 선택하였습니다.

## 환경 설정
**요구 사항**
Python 3.10+
Google Colab (T4 GPU 권장, 학습 시)
Google Drive 마운트 (모델 가중치 및 데이터 저장)

## 패키지 설치
bash/ pip install -r requirements.txt
주요 패키지: torch==2.12.0, transformers==5.9.0, openai-whisper==20250625, scikit-learn==1.7.2, gradio

## 재현 가이드
모든 노트북은 Google Colab에서 순서대로 실행합니다.
Google Drive에 presentation_data/ 폴더를 생성하고 데이터와 모델을 저장합니다.

### step 0 데이터셋 준비
AI Hub에서 아래 데이터셋을 다운로드합니다. (회원가입 및 신청 필요).
- 한국어 강의 음성 https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=115
- 한국어 음성 https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=123
- 영어 말하기 평가 https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71418
다운로드 후 Google Drive에 아래와 같이 배치합니다.
MyDrive/
└── presentation_data/
    ├── KLEC/          # 한국어 강의 음성 전사 텍스트 (.txt, .json)
    ├── KsponSpeech/   # 한국어 대화 음성 전사 텍스트
    └── english_eval/  # 영어 말하기 평가 데이터

### step 1 : EDA (01_EDA.ipynb)
- KsponSpeech 필러워드 비율 분포 분석 → 한국어 일상 발화 기준값 산출
- 영어 말하기 평가 점수(rater_final) 분포 확인 → 레이블 설계 방향 수립
- 주요 출력 : 필러워드 비율 분포 그래프, 영어 평가 점수 분포 그래프

### step 2 : 전처리 및 레이블링 (02_preprocessing.ipynb)
- KLEC 데이터에서 10,000개 발화를 샘플링하고 피처를 추출합니다.

실행 전 경로 확인
KLEC_DIR = '/content/drive/MyDrive/presentation_data/KLEC/'
OUTPUT_DIR = '/content/drive/MyDrive/presentation_data/'
SAMPLE_SIZE = 10000

- 처리 순서:
전사 텍스트 로드 및 clean_text 정제
5개 수치형 피처 추출 (extract_features)
filler_ratio 중앙값 기준 Good/Poor 레이블 부여
stratify 분할 → train.csv / val.csv / test.csv 저장

- 주의 : 베이스라인용으로는 filler_ratio, filler_count를 제외한 3개 피처만 사용합니다 (데이터 누수 방지).

### step 3 : 베이스라인 (03_baseline.ipynb)
FEATURE_COLS = ['total_words', 'vocab_diversity', 'avg_word_len']  # filler 피처 제외
- Logistic Regression / Random Forest 학습 및 평가
- Random Forest 피처 중요도 시각화

### step 4 : KLUE-RoBERTa 단독 (04_main_model.ipynb)
MODEL_NAME = 'klue/roberta-small'
LR = 2e-5
BATCH_SIZE = 32
EPOCHS = 5
MAX_LEN = 128

- clean_text → 토크나이징 → [CLS] 벡터 → Linear 분류기
- 학습 곡선(Train/Val Loss, F1) 시각화

### step 5 : FeatureConcat 모델 (05_feature_concat_model.ipynb → 05_feature_concat_model2.ipynb)
- 최종 하이퍼파라미터
CLASS_WEIGHT = torch.tensor([2.5, 1.0])   # Poor:Good
EARLY_STOPPING_PATIENCE = 2               # val_loss 기준
LR = 2e-5
BATCH_SIZE = 32
MAX_EPOCHS = 5

- 실험 순서:
05_feature_concat_model.ipynb : Concat 구조 도입 (5K 데이터, weight 없음 → weight=2.0)
05_feature_concat_model2.ipynb : 10K 확장 + Early Stopping + weight 2.0/2.5 비교

- 모델 저장 경로 확인
SAVE_PATH = '/content/drive/MyDrive/presentation_data/best_model_concat.pt'

### step 6 : 추론 검증 (06_Inference.ipynb)
저장된 best_model_concat.pt를 로드하고 inference_utils.py를 이용해 추론 파이프라인을 검증합니다.

from inference_utils import load_model_and_scaler, run_inference

model, tokenizer, scaler = load_model_and_scaler()

- 텍스트 직접 입력
result = run_inference("발표 텍스트를 여기에 입력하세요", model, tokenizer, scaler)
print(result['prediction'])   # Good / Poor
print(result['feedback'])     # 항목별 피드백

- 음성 파일 입력 
from inference_utils import stt_from_audio

text, duration = stt_from_audio("your_audio.wav")
result = run_inference(text, model, tokenizer, scaler, duration_seconds=duration)

### step 7 : Gradio 대시 보드 (07_dashboard.ipynb)
- Colab에서 실행 시 share=True로 공개 URL 생성
demo.launch(share=True)

- 음성 파일 업로드 또는 텍스트 직접 입력 → Good/Poor 판정 + 3항목 피드백 확인

### 경로 설정 (inference_utils.py)
로컬 또는 다른 환경에서 사용할 경우 파일 상단의 경로를 수정합니다.
MODEL_PATH = '/content/drive/MyDrive/presentation_data/best_model_concat.pt'
TRAIN_CSV  = '/content/drive/MyDrive/presentation_data/train.csv'