"""
inference_utils.py
발표 스킬 자동 분석 시스템 — 공통 유틸리티
모델 정의, feature 추출, 추론, 피드백 생성 함수 모음
"""

import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from sklearn.preprocessing import StandardScaler

# ── 경로 설정 ──────────────────────────────────────────────────
MODEL_PATH = '/content/drive/MyDrive/presentation_data/best_model_concat.pt'
TRAIN_CSV  = '/content/drive/MyDrive/presentation_data/train.csv'
# ──────────────────────────────────────────────────────────────

DEVICE       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME   = 'klue/roberta-small'
NUMERIC_COLS = ['total_words', 'filler_count', 'filler_ratio', 'vocab_diversity', 'avg_word_len']
FILLER_TAGS  = ['아/', '어/', '음/', '그/', '뭐/', '저/', '이/', 'n/']

# WPM 기준
WPM_TOO_SLOW  = 200
WPM_SLOW      = 250
WPM_FAST      = 350
WPM_TOO_FAST  = 400


# ── 모델 정의 ──────────────────────────────────────────────────
class RoBERTaWithFeatures(nn.Module):
    def __init__(self, model_name, num_features=5, num_labels=2, dropout=0.1):
        super().__init__()
        self.roberta    = AutoModel.from_pretrained(model_name)
        hidden_size     = self.roberta.config.hidden_size  # 768
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, num_features, labels=None):
        outputs    = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)
        combined   = torch.cat([cls_output, num_features], dim=1)
        logits     = self.classifier(combined)
        loss = None
        if labels is not None:
            weight = torch.tensor([2.5, 1.0]).to(labels.device)
            loss   = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return loss, logits


# ── 모델 & 토크나이저 & 스케일러 로드 ─────────────────────────
def load_model_and_scaler():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = RoBERTaWithFeatures(MODEL_NAME).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    df_train = pd.read_csv(TRAIN_CSV)
    scaler   = StandardScaler()
    scaler.fit(df_train[NUMERIC_COLS])

    print(f'✅ 모델 로드 완료 (device: {DEVICE})')
    print('✅ 스케일러 복원 완료')
    return model, tokenizer, scaler


# ── Whisper STT ────────────────────────────────────────────────
_whisper_model = None

def get_whisper_model(size='medium'):
    global _whisper_model
    if _whisper_model is None:
        import whisper
        print(f'Whisper {size} 모델 로딩 중...')
        _whisper_model = whisper.load_model(size)
        print('✅ Whisper 로드 완료')
    return _whisper_model


def stt_from_audio(audio_path: str) -> tuple:
    """음성 파일 → (전사 텍스트, 길이(초))"""
    wm       = get_whisper_model()
    result   = wm.transcribe(audio_path, language='ko', verbose=False)
    text     = result['text']
    segments = result.get('segments', [])
    duration = segments[-1]['end'] if segments else 0.0
    return text, duration


# ── Feature 추출 ───────────────────────────────────────────────
def clean_text(text: str) -> str:
    """학습(02_preprocessing)과 동일한 방식으로 텍스트 정제"""
    words = text.strip().split()
    words = [w for w in words if not any(w.startswith(tag) or w == tag[:-1] for tag in FILLER_TAGS)]
    words = [w for w in words if '/' not in w]
    words = [w for w in words if not re.match(r'^\((?:\d+번)\)?$', w)]
    words = [w for w in words if w.strip()]
    return ' '.join(words)


def extract_features(text: str) -> dict:
    """학습 때와 동일한 5개 숫자 feature 계산"""
    words       = text.strip().split()
    total_words = len(words)

    filler_count = sum(1 for w in words if any(w.startswith(tag) or w == tag[:-1] for tag in FILLER_TAGS))
    filler_ratio = filler_count / total_words if total_words > 0 else 0.0

    cleaned     = clean_text(text)
    clean_words = cleaned.split()
    vocab_diversity = len(set(clean_words)) / len(clean_words) if clean_words else 0.0
    avg_word_len    = float(np.mean([len(w) for w in clean_words])) if clean_words else 0.0

    return {
        'total_words':     total_words,
        'filler_count':    filler_count,
        'filler_ratio':    round(filler_ratio, 4),
        'vocab_diversity': round(vocab_diversity, 4),
        'avg_word_len':    round(avg_word_len, 4),
        'clean_text':      cleaned,
    }


def calc_wpm(text: str, duration_seconds: float) -> float:
    """피드백 참고용 WPM 계산 (모델 입력 아님)"""
    total_words = len(text.strip().split())
    if duration_seconds and duration_seconds > 0:
        return round(total_words / (duration_seconds / 60.0), 2)
    return 0.0


# ── 피드백 생성 ────────────────────────────────────────────────
def generate_feedback(features: dict, wpm: float = 0.0) -> dict:
    fb = {}
    fr = features['filler_ratio']
    fc = features['filler_count']
    tw = features['total_words']

    # 1) 필러워드 피드백 (3단계)
    if fr < 0.03:
        fb['filler'] = (
            f"✅ 필러워드 비율 {fr*100:.1f}% — 매우 양호합니다. "
            f"({fc}개 / 전체 {tw}어절)"
        )
    elif fr < 0.07:
        target = int(tw * 0.03)
        reduce = fc - target
        fb['filler'] = (
            f"🟡 필러워드 비율 {fr*100:.1f}% — 보통 수준입니다. "
            f"({fc}개 / 전체 {tw}어절) "
            f"약 {reduce}개 줄이면 3% 이하가 됩니다."
        )
    else:
        target = int(tw * 0.03)
        reduce = fc - target
        fb['filler'] = (
            f"🔴 필러워드 비율 {fr*100:.1f}% — 개선이 필요합니다. "
            f"({fc}개 / 전체 {tw}어절) "
            f"약 {reduce}개 줄여야 3% 이하가 됩니다. "
            f"발표 전 스크립트를 소리내어 연습하며 필러워드를 의식적으로 제거해보세요."
        )

    # 2) 어휘 다양성 피드백 (3단계)
    vd = features['vocab_diversity']
    if vd > 0.70:
        fb['vocab'] = f"✅ 어휘 다양성(TTR) {vd:.2f} — 풍부한 어휘를 사용하고 있습니다."
    elif vd > 0.50:
        fb['vocab'] = (
            f"🟡 어휘 다양성(TTR) {vd:.2f} — 보통 수준입니다. "
            f"반복되는 단어를 유사어나 다양한 표현으로 바꿔보세요."
        )
    else:
        fb['vocab'] = (
            f"🔴 어휘 다양성(TTR) {vd:.2f} — 단어 반복이 많습니다. "
            f"같은 표현이 자주 등장해 청중이 지루함을 느낄 수 있습니다. "
            f"핵심 용어 외 일반 단어는 다양하게 바꿔보세요."
        )

    # 3) WPM 피드백 (5단계)
    if wpm == 0.0:
        fb['wpm'] = "ℹ️ WPM: 음성 길이 정보가 없어 계산할 수 없습니다."
    elif wpm < WPM_TOO_SLOW:
        fb['wpm'] = (
            f"🔴 말 속도 {wpm:.0f} WPM — 너무 느립니다. "
            f"한국어 발표 평균(250~350 어절/분)보다 많이 느려 청중이 집중하기 어렵습니다."
        )
    elif wpm < WPM_SLOW:
        fb['wpm'] = (
            f"🟡 말 속도 {wpm:.0f} WPM — 약간 느린 편입니다. "
            f"적정 속도(250~350 어절/분) 하한에 해당합니다. 조금 더 자연스럽게 이어서 말해보세요."
        )
    elif wpm <= WPM_FAST:
        fb['wpm'] = (
            f"✅ 말 속도 {wpm:.0f} WPM — 적정 속도입니다. "
            f"한국어 발표 평균(250~350 어절/분) 범위에 있습니다."
        )
    elif wpm <= WPM_TOO_FAST:
        fb['wpm'] = (
            f"🟡 말 속도 {wpm:.0f} WPM — 약간 빠른 편입니다. "
            f"강조할 부분에서 의식적으로 속도를 줄여보세요."
        )
    else:
        fb['wpm'] = (
            f"🔴 말 속도 {wpm:.0f} WPM — 너무 빠릅니다. "
            f"한국어 발표 평균(250~350 어절/분)을 크게 초과했습니다. "
            f"문장 사이에 짧은 멈춤을 추가하고 호흡을 조절해보세요."
        )

    return fb


# ── 추론 ───────────────────────────────────────────────────────
def run_inference(text: str, model, tokenizer, scaler, duration_seconds: float = 0.0) -> dict:
    features = extract_features(text)

    numeric_raw = np.array([[
        features['total_words'],
        features['filler_count'],
        features['filler_ratio'],
        features['vocab_diversity'],
        features['avg_word_len']
    ]])
    numeric_scaled = scaler.transform(numeric_raw)

    encoding = tokenizer(
        features['clean_text'],
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids      = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    numeric_tensor = torch.tensor(numeric_scaled, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        _, logits = model(input_ids, attention_mask, numeric_tensor)
        probs     = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(probs))
    pred_label = 'Good' if pred_idx == 1 else 'Poor'
    wpm        = calc_wpm(text, duration_seconds)
    feedback   = generate_feedback(features, wpm)

    return {
        'prediction': pred_label,
        'prob_good':  round(float(probs[1]), 4),
        'prob_poor':  round(float(probs[0]), 4),
        'features':   features,
        'wpm':        wpm,
        'feedback':   feedback,
    }
