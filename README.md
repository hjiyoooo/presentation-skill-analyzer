# 🎙️ Presentation Skill Analyzer

본 프로젝트는 한국어 강의 및 발표 데이터를 기반으로 발표 품질을 평가(Good/Poor)하고, 발표자에게 발표 분석 및 개선 피드백을 제공하는 ML 시스템이다.

## 주요 기능
1. **Audio STT Pipeline**: OpenAI Whisper를 통한 발표 음성의 실시간 텍스트 변환 및 전처리 (`clean_text`)
2. **Hybrid Classification Model**: KLUE-RoBERTa의 언어 컨텍스트 벡터([CLS])와 5가지 수치형 발표 피처(필러워드 비율, 어휘 다양성, 단어 길이 등)를 결합한 Concat 분류기 모델
3. **Interactive Dashboard**: Gradio를 활용한 유저 친화적 웹 인터페이스 대시보드 (`07_dashboard.ipynb`)

## 모델 아키텍쳐
- **Text Embedding**: `klue/roberta-small` (768 Dimensions)
- **Numeric Features (5D)**: Total Words, Filler Count, Filler Ratio, Vocab Diversity, Avg Word Length
- **Concat Layer**: 768D + 5D = 773D $\rightarrow$ Linear Classifier $\rightarrow$ Binary Output (Good / Poor)

## 실험 및 성능 결과
- **Test F1-Score** : 0.7067
- **Accuracy** : 67.3%
